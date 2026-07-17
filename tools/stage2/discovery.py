"""Discover Stage 1 cases and freeze eligible cases into a manifest."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import REPO_ROOT, git_revision, hash_files, read_json, utc_now, write_json


ROOT_TOOLS = REPO_ROOT / "tools"
if str(ROOT_TOOLS) not in sys.path:
    sys.path.insert(0, str(ROOT_TOOLS))
import _stage1_common as S1  # noqa: E402


GROUND_TRUTH_LOGS = (
    "build_result.json",
    "run_result.json",
    "verify_result.json",
    "perf_result.json",
)
FINGERPRINT_INPUTS = (
    "metadata.json",
    "original/main.cu",
    "original/CMakeLists.txt",
    "tests/verify.py",
)


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    path: str
    category: str
    domain: str
    difficulty: str
    fingerprint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "path": self.path,
            "category": self.category,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "fingerprint": self.fingerprint,
        }


def _log_passes(case: S1.CaseInfo, filename: str) -> bool:
    path = case.path / "logs" / filename
    if not path.is_file():
        return False
    try:
        return read_json(path).get("status") == "pass"
    except (OSError, ValueError):
        return False


def is_cuda_ground_truth(case: S1.CaseInfo) -> bool:
    return case.pool == "benchmark" and all(_log_passes(case, name) for name in GROUND_TRUTH_LOGS)


def fingerprint_case(case: S1.CaseInfo) -> str:
    return fingerprint_case_path(case.path, case.case_id)


def fingerprint_case_path(case_path: Path, case_id: str) -> str:
    paths = [case_path / rel for rel in FINGERPRINT_INPUTS]
    missing = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{case_id} is missing fingerprint inputs: {', '.join(missing)}")
    return hash_files(paths)


def discover_eligible_cases() -> list[FrozenCase]:
    frozen: list[FrozenCase] = []
    for case in S1.iter_cases():
        if not is_cuda_ground_truth(case):
            continue
        meta = case.metadata
        frozen.append(
            FrozenCase(
                case_id=case.case_id,
                path=case.relpath,
                category=str(meta.get("category") or case.category),
                domain=str(meta.get("domain") or "unknown"),
                difficulty=str(meta.get("difficulty") or "unknown"),
                fingerprint=fingerprint_case(case),
            )
        )
    return sorted(frozen, key=lambda item: (item.category, item.case_id.lower()))


def build_manifest(dataset_id: str | None = None) -> dict[str, Any]:
    cases = discover_eligible_cases()
    return {
        "schema_version": 1,
        "dataset_id": dataset_id or f"cuda-verified-{len(cases)}-v1",
        "dataset_commit": git_revision(),
        "generated_at": utc_now(),
        "eligibility": {
            "layout": "benchmark/cases/<category>/<case>",
            "required_log_status": {name: "pass" for name in GROUND_TRUTH_LOGS},
        },
        "case_count": len(cases),
        "cases": [case.as_dict() for case in cases],
    }


def write_manifest(
    path: Path, expected_count: int | None = None, dataset_id: str | None = None
) -> dict[str, Any]:
    manifest = build_manifest(dataset_id=dataset_id)
    if expected_count is not None and manifest["case_count"] != expected_count:
        raise ValueError(
            f"expected {expected_count} eligible cases, found {manifest['case_count']}; "
            "the existing frozen manifest was not changed"
        )
    write_json(path, manifest)
    return manifest


def load_manifest(path: Path, check_files: bool = True) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema in {path}")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"manifest cases must be an array: {path}")
    if manifest.get("case_count") != len(cases):
        raise ValueError(f"manifest case_count does not match cases array: {path}")
    seen: set[str] = set()
    for record in cases:
        if not isinstance(record, dict) or not record.get("case_id") or not record.get("path"):
            raise ValueError(f"invalid case record in {path}")
        case_id = str(record["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id!r} in {path}")
        seen.add(case_id)
        if check_files:
            case_path = (REPO_ROOT / str(record["path"])).resolve()
            if not case_path.is_dir() or REPO_ROOT.resolve() not in case_path.parents:
                raise ValueError(f"manifest case path is missing or unsafe: {case_path}")
            actual_fingerprint = fingerprint_case_path(case_path, case_id)
            if record.get("fingerprint") != actual_fingerprint:
                raise ValueError(
                    f"case {case_id!r} has drifted from the frozen manifest; regenerate a new dataset version"
                )
    return manifest


def manifest_case_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["case_id"]): record for record in manifest["cases"]}


def audit_frozen_manifest(path: Path) -> dict[str, Any]:
    """Recheck the immutable inputs and all four NVIDIA ground-truth gates."""
    manifest = load_manifest(path, check_files=True)
    failures: list[dict[str, Any]] = []
    for record in manifest["cases"]:
        case_path = REPO_ROOT / str(record["path"])
        for filename in GROUND_TRUTH_LOGS:
            log_path = case_path / "logs" / filename
            try:
                status = read_json(log_path).get("status")
            except (OSError, ValueError) as exc:
                failures.append(
                    {"case_id": record["case_id"], "gate": filename, "status": "invalid", "detail": str(exc)}
                )
                continue
            if status != "pass":
                failures.append(
                    {"case_id": record["case_id"], "gate": filename, "status": status}
                )
    repository_case_count = sum(1 for case in S1.iter_cases() if case.pool == "benchmark")
    excluded_case_count = max(0, repository_case_count - int(manifest["case_count"]))
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_commit": manifest["dataset_commit"],
        "case_count": manifest["case_count"],
        "required_gates_per_case": len(GROUND_TRUTH_LOGS),
        "checked_gates": manifest["case_count"] * len(GROUND_TRUTH_LOGS),
        "failure_count": len(failures),
        "failures": failures,
        "repository_case_count": repository_case_count,
        "excluded_case_count": excluded_case_count,
        "all_repository_cases_admitted": excluded_case_count == 0,
        "warning": (
            f"{excluded_case_count} repository cases are outside this frozen NVIDIA-validated dataset"
            if excluded_case_count
            else None
        ),
        "ready": not failures,
    }
