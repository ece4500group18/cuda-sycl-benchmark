#!/usr/bin/env python3
"""Shared helpers for Stage 1 CUDA dataset validation.

The root ``tools/`` scripts are CUDA-ground-truth oriented. They discover both
the original pilot cases and the collection-phase adapted cases, but they do
not run or evaluate SYCL migration.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

STANDARD_REQUIRED_FILES = [
    "metadata.json",
    "README.md",
    "original/main.cu",
    "original/CMakeLists.txt",
    "original/README.md",
    "tests/verify.py",
]

STANDARD_OPTIONAL_DIRS = [
    "input",
    "expected",
    "output",
    "logs",
    "migrated",
]

STAGE1_METADATA_FIELDS = [
    "id",
    "name",
    "domain",
    "category",
    "difficulty",
    "source_project",
    "source_url",
    "license",
    "adaptation_type",
    "cuda_features",
    "description",
    "input_description",
    "output_description",
    "verification",
    "build",
    "run",
    "performance",
    "hardware",
    "status",
    "owner",
    "notes",
]

COMPAT_METADATA_FIELDS = [
    "case_id",
    "name",
    "category",
    "source",
    "cuda_features",
    "input",
    "build",
    "run",
    "correctness",
    "status",
    "notes",
]

SUMMARY_COLUMNS = [
    "case_name",
    "domain",
    "difficulty",
    "source",
    "source_type",
    "license",
    "CUDA_features",
    "build_ready",
    "run_ready",
    "verify_ready",
    "declared_status",
    "actual_verify_status",
    "actual_perf_status",
    "actual_status",
    "verify_log_path",
    "performance_log_path",
    "failure_reason",
    "notes",
]


@dataclass(frozen=True)
class CaseInfo:
    path: Path
    metadata_path: Path
    relpath: str
    case_id: str
    pool: str
    collection_slug: str | None
    category: str
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def metadata_paths() -> list[Path]:
    ignored = {".git", "__pycache__", "original/build", "build", "build_sycl"}
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("metadata.json"):
        rel = path.relative_to(REPO_ROOT)
        rel_posix = rel.as_posix()
        if any(part in ignored for part in rel.parts):
            continue
        if rel_posix.startswith("tools/"):
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(REPO_ROOT).as_posix().lower())


def classify_case(case_dir: Path, metadata: dict[str, Any]) -> tuple[str, str | None, str]:
    rel_parts = case_dir.relative_to(REPO_ROOT).parts
    if len(rel_parts) >= 4 and rel_parts[:2] == ("benchmark", "cases"):
        # Unified layout: benchmark/cases/<collection-category>/<case>/
        return "benchmark", rel_parts[2], rel_parts[2]
    if len(rel_parts) >= 4 and rel_parts[:2] == ("pilot_benchmark", "cases"):
        # Legacy pilot layout (kept for compatibility with older checkouts).
        return "pilot_benchmark", None, rel_parts[2]
    if len(rel_parts) >= 5 and rel_parts[:2] == ("benchmark", "collection"):
        return "collection", rel_parts[2], str(metadata.get("category", rel_parts[2]))
    return "unknown", None, str(metadata.get("category", "other"))


def iter_cases(case_filter: str | None = None) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    for meta_path in metadata_paths():
        case_dir = meta_path.parent
        try:
            meta = read_json(meta_path)
        except Exception:
            meta = {}
        pool, collection_slug, category = classify_case(case_dir, meta)
        case_id = str(meta.get("case_id") or meta.get("id") or case_dir.name)
        relpath = case_dir.relative_to(REPO_ROOT).as_posix()
        if case_filter and case_filter not in {case_id, case_dir.name, relpath}:
            continue
        cases.append(
            CaseInfo(
                path=case_dir,
                metadata_path=meta_path,
                relpath=relpath,
                case_id=case_id,
                pool=pool,
                collection_slug=collection_slug,
                category=category,
                metadata=meta,
            )
        )
    return cases


def ensure_case_dirs(case: CaseInfo, gitkeep: bool = False) -> None:
    for rel in STANDARD_OPTIONAL_DIRS:
        directory = case.path / rel
        directory.mkdir(parents=True, exist_ok=True)
        if gitkeep:
            gitkeep_path = directory / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.write_text("", encoding="utf-8")


def existing_files(case: CaseInfo) -> set[str]:
    return {
        rel
        for rel in STANDARD_REQUIRED_FILES
        if (case.path / rel).is_file()
    }


def existing_dirs(case: CaseInfo) -> set[str]:
    return {
        rel
        for rel in STANDARD_OPTIONAL_DIRS
        if (case.path / rel).is_dir()
    }


def source_url(meta: dict[str, Any]) -> str:
    source = meta.get("source", {})
    if isinstance(source, dict):
        return str(source.get("url", ""))
    return str(meta.get("source_url", ""))


def source_license(meta: dict[str, Any]) -> str:
    source = meta.get("source", {})
    if isinstance(source, dict):
        return str(source.get("license", ""))
    return str(meta.get("license", ""))


def source_project(meta: dict[str, Any]) -> str:
    explicit = meta.get("source_project")
    if explicit:
        return str(explicit)
    url = source_url(meta)
    marker = "github.com/"
    if marker in url:
        tail = url.split(marker, 1)[1]
        parts = [p for p in tail.split("/") if p]
        if len(parts) >= 2:
            return "/".join(parts[:2])
    source = meta.get("source", {})
    if isinstance(source, dict) and source.get("type"):
        source_type = str(source["type"])
        if source_type == "authored":
            return "ece4500group18/cuda-sycl-benchmark"
        return source_type
    return ""


def normalized_status_from_legacy(status: Any) -> str:
    """Summarize legacy metadata.status without treating it as ground truth."""
    if isinstance(status, str):
        return status
    if isinstance(status, dict):
        cuda_perf = status.get("cuda_performance")
        cuda_verify = status.get("cuda_verify")
        cuda_run = status.get("cuda_run")
        cuda_compile = status.get("cuda_compile")
        if cuda_perf == "pass":
            return "perf_ready"
        if cuda_verify == "pass":
            return "verified"
        if cuda_run == "pass":
            return "run_ready"
        if cuda_compile == "pass":
            return "build_ready"
        if any(str(v).startswith("fail") for v in status.values()):
            return "needs_review"
        if any(str(v).startswith("skipped") for v in status.values()):
            return "prepared"
    return "raw"


def declared_status(case: CaseInfo) -> str:
    return normalized_status_from_legacy(case.metadata.get("status", {}))


def adaptation_type(meta: dict[str, Any]) -> str:
    source = meta.get("source", {})
    value = meta.get("adaptation_type")
    if value:
        return str(value)
    if isinstance(source, dict):
        source_type = str(source.get("type", ""))
        if source_type in {"copied", "adapted", "simplified", "extracted"}:
            return source_type
        if source_type:
            return "adapted" if source_type != "authored" else "simplified"
    return "adapted"


def source_type(meta: dict[str, Any]) -> str:
    source = meta.get("source", {})
    if isinstance(source, dict) and source.get("type"):
        return str(source["type"])
    if meta.get("source_url"):
        return str(meta.get("adaptation_type") or "adapted")
    return "unknown"


def infer_domain(case: CaseInfo) -> str:
    if case.metadata.get("domain"):
        return str(case.metadata["domain"])
    name = case.case_id.lower()
    category = case.category.lower()
    if category == "hpc":
        return "hpc"
    if category == "library_api":
        return "library_api"
    if category in {"easy", "medium"}:
        return "cuda_primitive"
    if case.collection_slug == "stencil-convolution":
        image_terms = [
            "resize", "crop", "flip", "rotate", "color", "normalize",
            "transpose", "paste", "warp", "bbox", "box", "image", "pool",
        ]
        if any(term in name for term in image_terms):
            return "image_processing"
        return "modern_ml"
    if category in {"ai", "modern_ml", "ml"}:
        return "modern_ml"
    return "other"


def infer_difficulty(case: CaseInfo) -> str:
    if case.metadata.get("difficulty"):
        value = str(case.metadata["difficulty"])
        return "unsupported" if value == "unsupported/edge" else value
    category = case.category.lower()
    features = ";".join(str(x).lower() for x in case.metadata.get("cuda_features", []))
    if case.collection_slug:
        return "hard"
    if category == "easy":
        return "easy"
    if category in {"medium", "hpc"}:
        if any(key in features for key in ["atomic", "stream", "graph", "cublas", "cufft"]):
            return "hard"
        return "medium"
    if category in {"ai", "library_api"}:
        return "hard"
    return "medium"


def build_command(meta: dict[str, Any]) -> str:
    build = meta.get("build", {})
    if isinstance(build, dict):
        return str(build.get("cuda_build_command") or build.get("command") or "")
    return ""


def run_command(meta: dict[str, Any]) -> str:
    run = meta.get("run", {})
    if isinstance(run, dict):
        return str(run.get("cuda_run_command") or run.get("command") or "")
    return ""


def verification_type(meta: dict[str, Any]) -> str:
    verification = meta.get("verification", {})
    if isinstance(verification, dict) and verification.get("type"):
        return str(verification["type"])
    correctness = meta.get("correctness", {})
    if isinstance(correctness, dict):
        method = str(correctness.get("method", ""))
        metric = str(correctness.get("metric", ""))
        if method == "cpu_reference":
            return "cpu_reference"
        if metric == "exact":
            return "exact"
        if metric in {"max_abs_error", "max_rel_error"}:
            return "tolerance"
        if method in {"residual_norm", "analytic_reference", "statistical"}:
            return "invariant"
    return ""


def case_log(case: CaseInfo, filename: str) -> dict[str, Any] | None:
    path = case.path / "logs" / filename
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def case_log_status(case: CaseInfo, filename: str) -> str | None:
    data = case_log(case, filename)
    if data is None:
        return None
    status = data.get("status")
    return str(status) if status is not None else None


def log_relpath(case: CaseInfo, filename: str) -> str:
    path = case.path / "logs" / filename
    return path.relative_to(REPO_ROOT).as_posix() if path.is_file() else ""


def actual_verify_status(case: CaseInfo) -> str:
    return case_log_status(case, "verify_result.json") or "not_run"


def actual_perf_status(case: CaseInfo) -> str:
    return case_log_status(case, "perf_result.json") or "not_run"


def failure_reason(case: CaseInfo) -> str:
    reasons: list[str] = []
    for filename, label in (("verify_result.json", "verify"), ("perf_result.json", "perf")):
        data = case_log(case, filename)
        if not data:
            continue
        status = data.get("status")
        if status == "pass":
            continue
        message = str(data.get("message") or "")
        failures = data.get("failures")
        if not message and failures:
            message = str(failures)
        if not message and data.get("returncode") is not None:
            message = f"returncode={data.get('returncode')}"
        reasons.append(f"{label}:{status}" + (f" ({message})" if message else ""))
    return " | ".join(reasons)


def actual_status(case: CaseInfo) -> str:
    if actual_perf_status(case) == "pass":
        return "perf_ready"
    if actual_verify_status(case) == "pass":
        return "verified"
    ready = readiness(case)
    if ready["verify_ready"]:
        return "verify_ready"
    if ready["build_ready"]:
        return "build_ready"
    if ready["has_main_cu"]:
        return "prepared"
    return "raw"


def readiness(case: CaseInfo) -> dict[str, bool]:
    files = existing_files(case)
    has_build_cmd = bool(build_command(case.metadata))
    has_run_cmd = bool(run_command(case.metadata))
    has_verify_config = bool(case.metadata.get("verification") or case.metadata.get("correctness"))
    build_ready = all(rel in files for rel in STANDARD_REQUIRED_FILES) and has_build_cmd
    return {
        "has_main_cu": "original/main.cu" in files,
        "build_ready": build_ready,
        "run_ready": build_ready and has_run_cmd,
        "verify_ready": build_ready and has_run_cmd and "tests/verify.py" in files and has_verify_config,
        "perf_ready": actual_perf_status(case) == "pass",
    }


def audit_case(case: CaseInfo) -> dict[str, Any]:
    files = existing_files(case)
    dirs = existing_dirs(case)
    missing_files = [rel for rel in STANDARD_REQUIRED_FILES if rel not in files]
    missing_dirs = [rel for rel in STANDARD_OPTIONAL_DIRS if rel not in dirs]
    ready = readiness(case)
    status = actual_status(case)
    notes: list[str] = []
    if missing_files:
        notes.append("missing files: " + ", ".join(missing_files))
    if missing_dirs:
        notes.append("missing dirs: " + ", ".join(missing_dirs))
    if not source_license(case.metadata):
        notes.append("missing license")
    if not source_url(case.metadata):
        notes.append("missing source URL")
    if status in {"raw", "prepared", "needs_review", "rejected"} and case.metadata.get("notes"):
        notes.append(str(case.metadata.get("notes")))
    return {
        "case_name": case.case_id,
        "relpath": case.relpath,
        "pool": case.pool,
        "domain": infer_domain(case),
        "difficulty": infer_difficulty(case),
        "source": source_project(case.metadata),
        "source_type": source_type(case.metadata),
        "source_url": source_url(case.metadata),
        "license": source_license(case.metadata),
        "CUDA_features": ";".join(str(x) for x in case.metadata.get("cuda_features", [])),
        "build_ready": ready["build_ready"],
        "run_ready": ready["run_ready"],
        "verify_ready": ready["verify_ready"],
        "perf_ready": ready["perf_ready"],
        "declared_status": declared_status(case),
        "actual_verify_status": actual_verify_status(case),
        "actual_perf_status": actual_perf_status(case),
        "actual_status": status,
        "status": status,
        "verify_log_path": log_relpath(case, "verify_result.json"),
        "performance_log_path": log_relpath(case, "perf_result.json"),
        "failure_reason": failure_reason(case),
        "notes": " | ".join(notes),
        "extraction_fidelity": str(case.metadata.get("extraction_fidelity", case.metadata.get("adaptation_type", ""))),
        "extraction_notes": str(case.metadata.get("extraction_notes", case.metadata.get("notes", ""))),
        "description": str(case.metadata.get("description") or case.metadata.get("name") or case.case_id),
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "verification_type": verification_type(case.metadata),
        "build_command": build_command(case.metadata),
        "run_command": run_command(case.metadata),
    }


def summary_row(audit: dict[str, Any]) -> dict[str, str]:
    return {
        "case_name": str(audit["case_name"]),
        "domain": str(audit["domain"]),
        "difficulty": str(audit["difficulty"]),
        "source": str(audit["source"]),
        "source_type": str(audit["source_type"]),
        "license": str(audit["license"]),
        "CUDA_features": str(audit["CUDA_features"]),
        "build_ready": "yes" if audit["build_ready"] else "no",
        "run_ready": "yes" if audit["run_ready"] else "no",
        "verify_ready": "yes" if audit["verify_ready"] else "no",
        "declared_status": str(audit["declared_status"]),
        "actual_verify_status": str(audit["actual_verify_status"]),
        "actual_perf_status": str(audit["actual_perf_status"]),
        "actual_status": str(audit["actual_status"]),
        "verify_log_path": str(audit["verify_log_path"]),
        "performance_log_path": str(audit["performance_log_path"]),
        "failure_reason": str(audit["failure_reason"]),
        "notes": str(audit["notes"]),
    }


def stage1_metadata_path(case: CaseInfo) -> Path:
    return case.path / "metadata.stage1.json"


def source_url_for_stage1(case: CaseInfo) -> str:
    url = source_url(case.metadata)
    if url:
        return url
    source = case.metadata.get("source", {})
    if isinstance(source, dict) and source.get("type") == "authored":
        return "https://github.com/ece4500group18/cuda-sycl-benchmark"
    return ""


def stage1_verification(case: CaseInfo) -> dict[str, Any]:
    correctness = case.metadata.get("correctness", {})
    if not isinstance(correctness, dict):
        correctness = {}
    vtype = verification_type(case.metadata) or "cpu_reference"
    tol = correctness.get("tolerance")
    if tol is None:
        tol = 0.0 if vtype == "exact" else 1e-5
    if vtype == "cpu_reference":
        reference = "cpu"
    elif vtype == "exact":
        reference = "expected_file"
    elif vtype == "invariant":
        reference = "invariant"
    else:
        reference = "cpu"
    return {
        "type": vtype,
        "rtol": tol,
        "atol": tol,
        "reference": reference,
        "notes": str(correctness.get("method") or correctness.get("metric") or ""),
    }


def stage1_hardware(case: CaseInfo) -> dict[str, Any]:
    perf = case_log(case, "perf_result.json") or {}
    return {
        "gpu_model": perf.get("gpu_model"),
        "cuda_version": perf.get("cuda_version"),
        "driver_version": perf.get("driver_version"),
    }


def stage1_performance(case: CaseInfo) -> dict[str, Any]:
    perf = case_log(case, "perf_result.json") or {}
    input_meta = case.metadata.get("input", {})
    problem_size = input_meta.get("sizes") if isinstance(input_meta, dict) else ""
    return {
        "nvidia_baseline_required": True,
        "timing_available": perf.get("status") == "pass",
        "metric_type": perf.get("metric_type", "runtime"),
        "primary_metric": "ms",
        "problem_size": problem_size,
        "notes": perf.get("notes", "Runtime logs are written by tools/benchmark_case.py."),
    }


def build_stage1_metadata(case: CaseInfo) -> dict[str, Any]:
    meta = case.metadata
    input_meta = meta.get("input", {})
    input_desc = ""
    if isinstance(input_meta, dict):
        input_desc = (
            f"type={input_meta.get('type', '')}; "
            f"sizes={input_meta.get('sizes', [])}; seed={input_meta.get('seed', '')}"
        )
    return {
        "id": case.case_id,
        "name": str(meta.get("name") or case.case_id),
        "domain": infer_domain(case),
        "category": str(meta.get("category") or case.category),
        "difficulty": infer_difficulty(case),
        "source_project": source_project(meta),
        "source_url": source_url_for_stage1(case),
        "license": source_license(meta),
        "adaptation_type": adaptation_type(meta),
        "cuda_features": list(meta.get("cuda_features", [])),
        "description": str(meta.get("description") or meta.get("notes") or meta.get("name") or case.case_id),
        "input_description": input_desc,
        "output_description": "CUDA executable writes numeric output to output/cuda_output.txt.",
        "verification": stage1_verification(case),
        "build": {
            "system": "custom_nvcc",
            "requires_gpu": True,
            "requires_external_libs": list(meta.get("libraries", [])),
            "command": build_command(meta),
        },
        "run": {
            "command": run_command(meta),
            "output_files": ["output/cuda_output.txt"],
            "expected_runtime_sec": "",
        },
        "performance": stage1_performance(case),
        "hardware": stage1_hardware(case),
        "status": actual_status(case),
        "owner": "ece4500group18",
        "extraction_fidelity": str(meta.get("extraction_fidelity", adaptation_type(meta))),
        "extraction_notes": str(meta.get("extraction_notes", meta.get("notes", ""))),
        "notes": str(meta.get("notes", "")),
        "legacy_metadata": {
            "file": "metadata.json",
            "declared_status": declared_status(case),
        },
    }


def load_stage1_metadata(case: CaseInfo) -> dict[str, Any] | None:
    if all(field in case.metadata for field in STAGE1_METADATA_FIELDS) and isinstance(case.metadata.get("status"), str):
        return case.metadata
    path = stage1_metadata_path(case)
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def stage1_metadata_problems(doc: dict[str, Any] | None) -> list[str]:
    if doc is None:
        return ["metadata.stage1.json missing"]
    problems = [field for field in STAGE1_METADATA_FIELDS if field not in doc]
    if not isinstance(doc.get("status"), str):
        problems.append("status must be a Stage 1 string")
    for key in ("verification", "build", "run", "performance", "hardware"):
        if key in doc and not isinstance(doc[key], dict):
            problems.append(f"{key} must be an object")
    if not doc.get("source_url"):
        problems.append("source_url")
    if not doc.get("license"):
        problems.append("license")
    return problems


def validate_metadata(case: CaseInfo) -> dict[str, Any]:
    meta = case.metadata
    missing_compat = [field for field in COMPAT_METADATA_FIELDS if field not in meta]
    stage1_doc = load_stage1_metadata(case)
    stage1_problems = stage1_metadata_problems(stage1_doc)
    missing_stage1 = (
        [field for field in STAGE1_METADATA_FIELDS if stage1_doc is None or field not in stage1_doc]
    )
    source = meta.get("source", {})
    warnings: list[str] = []
    if isinstance(source, dict):
        for key in ("url", "license"):
            if not source.get(key):
                warnings.append(f"source.{key}")
    else:
        warnings.append("source object")
    command_problems: list[str] = []
    if not build_command(meta):
        command_problems.append("build.cuda_build_command")
    if not run_command(meta):
        command_problems.append("run.cuda_run_command")
    verification_problems = [] if verification_type(meta) else ["verification/correctness"]
    fail_reasons = missing_compat + command_problems + verification_problems
    return {
        "case_name": case.case_id,
        "relpath": case.relpath,
        "compat_valid": not fail_reasons,
        "strict_stage1_valid": not stage1_problems,
        "stage1_metadata_path": (
            stage1_metadata_path(case).relative_to(REPO_ROOT).as_posix()
            if stage1_metadata_path(case).is_file()
            else ""
        ),
        "missing_compat_fields": missing_compat,
        "missing_stage1_fields": missing_stage1,
        "problems": fail_reasons,
        "stage1_problems": stage1_problems,
        "warnings": warnings,
        "schema_note": (
            "legacy-compatible metadata"
            if not fail_reasons
            else "metadata needs repair"
        ),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def as_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        cells: list[str] = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, bool):
                value = "yes" if value else "no"
            cell = str(value).replace("|", "\\|").replace("\n", " ")
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def which(name: str) -> str | None:
    return shutil.which(name)


def find_vcvars64() -> str | None:
    if os.name != "nt":
        return None
    vswhere = which("vswhere")
    if not vswhere:
        pf86 = os.environ.get("ProgramFiles(x86)")
        if pf86:
            candidate = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
            if candidate.is_file():
                vswhere = str(candidate)
    if vswhere:
        try:
            out = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    r"VC\Auxiliary\Build\vcvars64.bat",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in out.stdout.splitlines():
                path = line.strip()
                if path and Path(path).is_file():
                    return path
        except Exception:
            pass
    for root in [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")]:
        if not root:
            continue
        vs_root = Path(root) / "Microsoft Visual Studio"
        if not vs_root.is_dir():
            continue
        for year in sorted(vs_root.iterdir(), reverse=True):
            if not year.is_dir():
                continue
            for edition in sorted(year.iterdir(), reverse=True):
                candidate = edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.is_file():
                    return str(candidate)
    return None


def cuda_toolchain_status() -> tuple[bool, str, str]:
    nvcc = which("nvcc")
    if not nvcc:
        return False, "skipped", "nvcc not found on PATH"
    if os.name == "nt" and which("cl") is None:
        vcvars = find_vcvars64()
        if not vcvars:
            return False, "skipped", "nvcc found, but cl.exe/vcvars64.bat was not found"
        return True, "pass", f"nvcc found; using Visual Studio environment {vcvars}"
    return True, "pass", f"nvcc found at {nvcc}"


def has_nvidia_gpu() -> bool:
    smi = which("nvidia-smi")
    if not smi:
        return False
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=15)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_model": None,
        "cuda_version": None,
        "driver_version": None,
        "compiler": None,
        "host": platform.platform(),
    }
    smi = which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                if parts:
                    info["gpu_model"] = parts[0]
                if len(parts) > 1:
                    info["driver_version"] = parts[1]
        except Exception:
            pass
    nvcc = which("nvcc")
    if nvcc:
        info["compiler"] = nvcc
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=15)
            text = out.stdout + out.stderr
            for line in text.splitlines():
                if "release " in line:
                    info["cuda_version"] = line.split("release ", 1)[1].split(",", 1)[0].strip()
                    break
        except Exception:
            pass
    return info


def shell_command(args: list[str | Path]) -> str:
    parts = [str(a) for a in args]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def python_command(*args: str | Path) -> str:
    return shell_command([sys.executable, *args])


def add_windows_nvcc_flags(command: str) -> str:
    if os.name != "nt":
        return command
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return command
    if not parts or Path(parts[0]).name.lower() != "nvcc":
        return command
    if "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" not in parts:
        parts.insert(1, "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
    if not any(p in {"-allow-unsupported-compiler", "--allow-unsupported-compiler"} for p in parts):
        parts.insert(1, "-allow-unsupported-compiler")
    return shell_command(parts)


def prepare_cuda_build_command(command: str) -> str:
    command = add_windows_nvcc_flags(command)
    if os.name != "nt" or which("cl") is not None:
        return command
    vcvars = find_vcvars64()
    if not vcvars:
        return command
    escaped = command.replace('"', r'\"')
    return f'cmd.exe /s /c ""{vcvars}" >nul && {escaped}"'


def normalize_executable_command(command: str, cwd: Path) -> str:
    if os.name != "nt" or not command.strip():
        return command
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return command
    if not parts:
        return command
    exe = parts[0]
    if Path(exe).suffix:
        return command
    candidate = Path(exe + ".exe")
    abs_candidate = candidate if candidate.is_absolute() else cwd / candidate
    if abs_candidate.is_file():
        parts[0] = str(candidate if candidate.is_absolute() else Path(".") / candidate)
        return shell_command(parts)
    return command


def run_logged(command: str, cwd: Path, log_path: Path, timeout: int = 600) -> tuple[int, str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    header = (
        f"# command: {command}\n"
        f"# cwd:     {cwd}\n"
        f"# time:    {utc_now()}\n"
        f"{'-' * 70}\n"
    )
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "") + f"\n[TIMEOUT after {timeout}s]\n"
        rc = 124
    elapsed = time.perf_counter() - start
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(output)
        fh.write(f"\n{'-' * 70}\n# exit code: {rc}\n# elapsed_sec: {elapsed:.6f}\n")
    return rc, output, elapsed


def benchmark_output_command(command: str, output_rel: str) -> str:
    replacements = [
        "output/cuda_output.txt",
        r"output\cuda_output.txt",
    ]
    for needle in replacements:
        if needle in command:
            repl = output_rel if "/" in needle else output_rel.replace("/", "\\")
            return command.replace(needle, repl)
    return f"{command} {output_rel}"


def runtime_summary_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def command_result(
    case: CaseInfo,
    status: str,
    command: str,
    elapsed_sec: float | None = None,
    returncode: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "case_name": case.case_id,
        "case_path": case.relpath,
        "status": status,
        "command": command,
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "message": message,
        "timestamp": utc_now(),
    }
