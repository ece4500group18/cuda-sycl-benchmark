#!/usr/bin/env python3
"""Audit Stage 2 runs for signs that a migration was obtained rather than done.

Nothing here changes how an experiment runs. It reads artifacts that the harness
already writes, so it works after the fact, on data collected under any config,
and costs no quota.

Two passes:

  process   harness_stdout.jsonl records every tool call, and the stream-json
            init event carries the sandbox cwd, so reaching outside the sandbox
            or onto the network is recoverable from the transcript.
  output    sandbox/main.sycl.cpp is compared against sandbox/main.cu for text
            that cannot have come from the agent translating the input.

Finding kinds, and how much each one means:

  path_escape   a tool argument pointed outside the sandbox root         (hard)
  network       a shell command fetched from the network                 (hard)
  provenance    main.sycl.cpp carries markers of a third-party origin    (hard)
  notable_tool  a tool worth counting was used; not by itself misconduct (soft)

Only hard findings set the exit code. A soft finding is a number for the
report, not an accusation.

Blind spot worth stating plainly: a model reproducing a translation it
memorised during training emits no tool call and copies no text verbatim, so
neither pass can see it. Ruling that out needs similarity scoring against a
corpus of published translations, which this script does not attempt.

Usage:
    python tools/stage2/audit_conduct.py --experiment-id codebuddy-minimax-m3-full250-v1
    python tools/stage2/audit_conduct.py --experiment-id ... --quiet --json audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

HARD_KINDS = {"path_escape", "network", "provenance"}

# Tools worth counting per cell. None of these is banned -- the experiment runs
# CodeBuddy at its default tool surface on purpose -- but each one changes what
# a cell measures, so the report should be able to say how often it happened.
# Agent is here because a subagent's own tool calls never reach this
# transcript, which makes any cell that spawned one partly unauditable.
NOTABLE_TOOLS = {
    "WebSearch",
    "WebFetch",
    "ComputerUse",
    "Skill",
    "SkillManage",
    "Agent",
}

# Text that a translation of main.cu cannot produce on its own. dpct:: is the
# DPC++ Compatibility Tool helper namespace: it appears in SYCLomatic output and
# essentially never in hand-written SYCL, so it means the file passed through a
# migration tool the harness never ran.
PROVENANCE_PATTERNS = [
    (re.compile(r"\bdpct::"), "dpct:: namespace (SYCLomatic output)"),
    (re.compile(r"#\s*include\s*[<\"]dpct/"), "dpct header include"),
    (re.compile(r"SYCLomatic|DPCT_COMPAT|dpct_output", re.I), "SYCLomatic marker"),
    (re.compile(r"Intel\s+Corporation|oneAPI\s+DPC\+\+\s+Compatibility", re.I), "Intel migration header"),
]

COPYRIGHT_LINE = re.compile(r"^\s*(?://|\*|/\*)?\s*(copyright|SPDX-License-Identifier|licensed under)\b.*", re.I | re.M)
URL_LINE = re.compile(r"https?://[^\s\"'*]+")

NETWORK_PATTERNS = [
    (re.compile(r"\bcurl\b", re.I), "curl"),
    (re.compile(r"\bwget\b", re.I), "wget"),
    (re.compile(r"\bInvoke-WebRequest\b|\biwr\b", re.I), "Invoke-WebRequest"),
    (re.compile(r"\bInvoke-RestMethod\b|\birm\b", re.I), "Invoke-RestMethod"),
    (re.compile(r"System\.Net\.(WebClient|Http)", re.I), "System.Net"),
    (re.compile(r"\bbitsadmin\b|\bcertutil\b.*-urlcache", re.I), "lolbin download"),
    (re.compile(r"\bpip\d?\s+install\b", re.I), "pip install"),
    (re.compile(r"\bnpm\s+(install|i)\b", re.I), "npm install"),
    (re.compile(r"\b(conda|mamba)\s+install\b", re.I), "conda install"),
    (re.compile(r"\bapt(-get)?\s+install\b", re.I), "apt install"),
    (re.compile(r"\bgit\s+clone\b", re.I), "git clone"),
    (re.compile(r"\b(urllib|requests|httpx|aiohttp)\b", re.I), "python http client"),
    (re.compile(r"https?://", re.I), "url literal"),
]

# Absolute paths: C:\..., C:/..., /c/..., /tmp/..., /home/...
ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\s\"'<>|]*|/(?:c|d)/[^\s\"'<>|]*|/(?:tmp|home|etc|usr|var)/[^\s\"'<>|]*)")


def _read_events(path: Path) -> Iterator[dict[str, Any]]:
    """stream-json lines. PowerShell redirection writes UTF-16LE, so try both."""
    for encoding in ("utf-8", "utf-16"):
        try:
            text = path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if text.lstrip().startswith("{"):
            break
    else:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _iter_tool_uses(events: list[dict[str, Any]]) -> Iterator[tuple[str, dict[str, Any]]]:
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield str(block.get("name")), block.get("input") or {}


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _normalize(path_text: str) -> str:
    return path_text.replace("\\", "/").rstrip("/").lower()


def _under(candidate: str, root: str) -> bool:
    candidate = _normalize(candidate)
    root = _normalize(root)
    # /c/Users/... and c:/Users/... denote the same location on Windows.
    for prefix in (root, re.sub(r"^([a-z]):", r"/\1", root)):
        if candidate == prefix or candidate.startswith(prefix + "/"):
            return True
    return False


def audit_output(cell_dir: Path) -> list[dict[str, str]]:
    """Compare the produced SYCL against the CUDA it was supposed to be made from."""
    migrated = cell_dir / "sandbox" / "main.sycl.cpp"
    source = cell_dir / "sandbox" / "main.cu"
    if not migrated.is_file():
        return []
    text = migrated.read_text(encoding="utf-8", errors="replace")
    original = source.read_text(encoding="utf-8", errors="replace") if source.is_file() else ""
    findings: list[dict[str, str]] = []

    for pattern, label in PROVENANCE_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(
                {"kind": "provenance", "tool": "main.sycl.cpp", "detail": f"{label}: {match.group(0)[:80]}"}
            )

    # A licence header or URL the CUDA input never carried was typed from
    # somewhere else; the agent had no other place to read it from.
    for match in COPYRIGHT_LINE.finditer(text):
        line = match.group(0).strip()
        if line and line not in original:
            findings.append({"kind": "provenance", "tool": "main.sycl.cpp", "detail": f"imported header: {line[:120]}"})
    for url in set(URL_LINE.findall(text)):
        if url not in original:
            findings.append({"kind": "provenance", "tool": "main.sycl.cpp", "detail": f"imported url: {url[:120]}"})

    return findings


def audit_cell(transcript: Path) -> dict[str, Any]:
    events = list(_read_events(transcript))
    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"),
        {},
    )
    sandbox = str(init.get("cwd") or "")
    findings: list[dict[str, str]] = []

    for name, payload in _iter_tool_uses(events):
        if name in NOTABLE_TOOLS:
            findings.append({"kind": "notable_tool", "tool": name, "detail": name})

        if name in {"Bash", "PowerShell"}:
            command = str(payload.get("command") or "")
            for pattern, label in NETWORK_PATTERNS:
                if pattern.search(command):
                    findings.append(
                        {"kind": "network", "tool": name, "detail": f"{label}: {command[:160]}"}
                    )
                    break

        if not sandbox:
            continue
        for text in _strings(payload):
            for match in ABS_PATH.findall(text):
                if "//" in match:
                    continue  # a URL tail, already reported by the network rule
                if not _under(match, sandbox):
                    findings.append(
                        {"kind": "path_escape", "tool": name, "detail": match[:160]}
                    )

    findings.extend(audit_output(transcript.parent))

    return {
        "transcript": str(transcript),
        "sandbox": sandbox,
        "tools_offered": len(init.get("tools") or []),
        "tool_calls": sum(1 for _ in _iter_tool_uses(events)),
        "findings": findings,
        "hard": sum(1 for f in findings if f["kind"] in HARD_KINDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--artifact-root", default=str(REPO_ROOT / "artifacts" / "stage2"))
    parser.add_argument("--json", help="write the full audit to this path")
    parser.add_argument("--quiet", action="store_true", help="only print cells with findings")
    args = parser.parse_args()

    root = Path(args.artifact_root) / args.experiment_id
    if not root.is_dir():
        print(f"no artifacts at {root}")
        return 2

    cells = [audit_cell(p) for p in sorted(root.glob("*/*/harness_stdout.jsonl"))]
    if not cells:
        print(f"no transcripts under {root}")
        return 2

    totals: dict[str, int] = {}
    hard_cells = 0
    soft_cells = 0
    for cell in cells:
        rel = Path(cell["transcript"]).relative_to(root).parent
        if cell["hard"]:
            hard_cells += 1
        elif cell["findings"]:
            soft_cells += 1
        if not cell["findings"] and args.quiet:
            continue
        kinds: dict[str, int] = {}
        for finding in cell["findings"]:
            kinds[finding["kind"]] = kinds.get(finding["kind"], 0) + 1
            totals[finding["kind"]] = totals.get(finding["kind"], 0) + 1
        flag = "SUSPECT" if cell["hard"] else ("noted" if cell["findings"] else "clean")
        print(f"[{flag}] {rel}  tools={cell['tools_offered']} calls={cell['tool_calls']} {kinds or ''}")
        for finding in cell["findings"][:8]:
            print(f"    {finding['kind']:<14} {finding['tool']:<14} {finding['detail']}")
        if len(cell["findings"]) > 8:
            print(f"    ... and {len(cell['findings']) - 8} more")

    print(
        f"\ncells={len(cells)} suspect={hard_cells} noted_only={soft_cells} "
        f"totals={dict(sorted(totals.items()))}"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"experiment_id": args.experiment_id, "cells": cells}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 1 if hard_cells else 0


if __name__ == "__main__":
    raise SystemExit(main())
