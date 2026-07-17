"""Combined Stage 1, Intel GPU, harness, model, and skill readiness gate."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from common import resolve_repo_path, utc_now
from adapters.codex import resolve_codex_cli
from discovery import audit_frozen_manifest
from doctor import probe_environment, probe_ssh_environment
from executor import LocalExecutor
from runner import load_experiment, plan_experiment, resolve_model_id


def _harness_check(harness: dict[str, Any]) -> dict[str, Any]:
    adapter = str(harness["adapter"])
    if adapter == "mock":
        return {"slug": harness["slug"], "adapter": adapter, "ready": True, "synthetic": True}
    if adapter == "codex":
        try:
            path, version = resolve_codex_cli(str(harness.get("command") or "codex"))
            return {
                "slug": harness["slug"], "adapter": adapter, "command": harness.get("command"),
                "path": path, "ready": True, "version": version,
            }
        except RuntimeError as exc:
            return {
                "slug": harness["slug"], "adapter": adapter, "command": harness.get("command"),
                "path": None, "ready": False, "error": str(exc),
            }
    command = harness.get("command")
    if adapter == "external_command":
        argv = harness.get("argv")
        command = argv[0] if isinstance(argv, list) and argv else None
    path = shutil.which(str(command)) if command else None
    check: dict[str, Any] = {
        "slug": harness["slug"], "adapter": adapter, "command": command,
        "path": path, "ready": path is not None,
    }
    if path:
        version_args = [str(item) for item in harness.get("version_args", ["--version"])]
        check["version"] = LocalExecutor().run([path, *version_args], Path.cwd(), 30).as_dict()
    return check


def preflight_experiment(
    experiment_path: Path,
    run_smoke: bool = True,
    require_intel: bool = True,
) -> dict[str, Any]:
    experiment = load_experiment(experiment_path)
    manifest_path = resolve_repo_path(str(experiment["dataset_manifest"]))
    dataset = audit_frozen_manifest(manifest_path)
    _, runs = plan_experiment(experiment)
    harnesses = [_harness_check(item) for item in experiment["harnesses"] if item.get("enabled", True)]
    models = []
    for model in experiment["models"]:
        if not model.get("enabled", True):
            continue
        resolved = resolve_model_id(model, require=False)
        alias_warning = resolved.lower() in {"opus", "sonnet", "haiku", "latest"}
        exact_required = bool(model.get("require_immutable_model_id"))
        models.append(
            {
                "slug": model["slug"], "provider": model["provider"],
                "requested_model_id": resolved,
                "ready": not resolved.startswith("${") and resolved != "<unset>"
                and not (exact_required and alias_warning),
                "immutable_id_warning": alias_warning,
                "immutable_id_required": exact_required,
            }
        )
    skills = []
    for condition in experiment["skill_conditions"]:
        value = condition.get("skill")
        path = resolve_repo_path(str(value)) if value else None
        ready = path is None or (path / "SKILL.md").is_file()
        skills.append({"slug": condition["slug"], "path": str(path) if path else None, "ready": ready})
    if experiment["executor"]["kind"] == "ssh":
        intel = probe_ssh_environment(experiment["executor"], run_smoke=run_smoke)
    else:
        intel = probe_environment(
            run_smoke=run_smoke,
            device_selector=str(experiment["executor"]["device_selector"]),
        )
    real_harnesses = [item for item in harnesses if not item.get("synthetic")]
    checks = {
        "dataset": dataset["ready"],
        "harnesses": all(item["ready"] for item in harnesses),
        "models": all(item["ready"] for item in models),
        "skills": all(item["ready"] for item in skills),
        "intel_gpu": intel["ready_for_intel_gpu_experiments"] if require_intel and real_harnesses else True,
    }
    return {
        "schema_version": 1, "timestamp": utc_now(),
        "experiment_id": experiment["experiment_id"], "planned_runs": len(runs),
        "dataset": dataset, "harnesses": harnesses, "models": models,
        "skill_conditions": skills, "intel_gpu": intel, "checks": checks,
        "ready": all(checks.values()),
        "blocking_reasons": [name for name, ready in checks.items() if not ready],
    }
