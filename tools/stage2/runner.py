"""Plan and execute case x harness x model x skill x repeat cells."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from adapters.base import HarnessAdapter, SessionContext
from adapters.claude_code import ClaudeCodeAdapter
from adapters.codex import CodexAdapter
from adapters.external_command import ExternalCommandAdapter
from adapters.mock import MockAdapter
from common import (
    DEFAULT_ARTIFACT_ROOT,
    REPO_ROOT,
    read_json,
    resolve_bash,
    resolve_repo_path,
    safe_component,
    utc_now,
    write_json,
)
from discovery import load_manifest, manifest_case_index
from executor import LocalExecutor
from remote import build_remote_config
from sandbox import create_sandbox, restore_fixed_wrappers
from verify import verify_case_output


@dataclass(frozen=True)
class PlannedRun:
    case: dict[str, Any]
    harness: dict[str, Any]
    model: dict[str, Any]
    skill_condition: dict[str, Any]
    repeat: int

    @property
    def run_id(self) -> str:
        return "--".join(
            (
                str(self.case["case_id"]),
                str(self.harness["slug"]),
                str(self.model["slug"]),
                str(self.skill_condition["slug"]),
                f"r{self.repeat}",
            )
        )


def load_experiment(path: Path) -> dict[str, Any]:
    experiment = read_json(path)
    required = {
        "schema_version",
        "experiment_id",
        "dataset_manifest",
        "case_ids",
        "harnesses",
        "models",
        "skill_conditions",
        "repeats",
        "budget",
        "executor",
        "reference_mode",
        "prompt_version",
    }
    missing = sorted(required - set(experiment))
    if missing:
        raise ValueError(f"experiment is missing: {', '.join(missing)}")
    if experiment["schema_version"] != 2:
        raise ValueError("only experiment schema_version 2 is supported")
    safe_component(str(experiment["experiment_id"]), "experiment_id")
    if int(experiment["repeats"]) < 1:
        raise ValueError("repeats must be positive")
    if experiment["reference_mode"] not in {"none", "cuda_output"}:
        raise ValueError("reference_mode must be none or cuda_output")
    if experiment["executor"].get("kind") not in {"local", "ssh"}:
        raise ValueError("executor.kind must be local or ssh")
    if int(experiment["executor"].get("gpu_slots", 0)) != 1:
        raise ValueError("gpu_slots must be 1 until deterministic multi-GPU scheduling is implemented")
    for collection, label in (
        (experiment["harnesses"], "harness"),
        (experiment["models"], "model"),
        (experiment["skill_conditions"], "skill condition"),
    ):
        if not isinstance(collection, list) or not collection:
            raise ValueError(f"{label}s must be a non-empty array")
        slugs = [safe_component(str(item["slug"]), f"{label} slug") for item in collection]
        if len(slugs) != len(set(slugs)):
            raise ValueError(f"duplicate {label} slug")
    return experiment


def resolve_model_id(model: dict[str, Any], require: bool = True) -> str:
    env_name = model.get("model_id_env")
    if env_name and os.environ.get(str(env_name)):
        return str(os.environ[str(env_name)])
    model_id = model.get("model_id")
    if model_id:
        return str(model_id)
    if require:
        raise ValueError(
            f"model {model.get('slug')!r} has no model_id; set {env_name}"
            if env_name
            else f"model {model.get('slug')!r} has no model_id"
        )
    return f"${{{env_name}}}" if env_name else "<unset>"


def _selected_ids(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    selected: set[str] = set()
    for value in values:
        selected.update(item.strip() for item in value.split(",") if item.strip())
    return selected


def _filter_records(
    records: list[dict[str, Any]], values: Iterable[str] | None, label: str
) -> list[dict[str, Any]]:
    records = [item for item in records if item.get("enabled", True)]
    selected = _selected_ids(values)
    if not selected:
        return records
    known = {str(item["slug"]) for item in records}
    missing = sorted(selected - known)
    if missing:
        raise ValueError(f"requested {label}s are not configured: {', '.join(missing)}")
    return [item for item in records if str(item["slug"]) in selected]


def _matrix_allows(experiment: dict[str, Any], harness: str, model: str) -> bool:
    matrix = experiment.get("matrix") or {}
    included = matrix.get("include")
    if not included:
        return True
    return any(
        str(item.get("harness")) == harness and str(item.get("model")) == model
        for item in included
    )


def plan_experiment(
    experiment: dict[str, Any],
    case_filters: Iterable[str] | None = None,
    harness_filters: Iterable[str] | None = None,
    model_filters: Iterable[str] | None = None,
    skill_filters: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[PlannedRun]]:
    manifest_path = resolve_repo_path(str(experiment["dataset_manifest"]))
    manifest = load_manifest(manifest_path)
    case_index = manifest_case_index(manifest)
    configured_cases = [str(item) for item in experiment["case_ids"]]
    unknown = sorted(set(configured_cases) - set(case_index))
    if unknown:
        raise ValueError(f"experiment references cases outside the manifest: {', '.join(unknown)}")

    selected_cases = _selected_ids(case_filters)
    if selected_cases:
        missing = sorted(selected_cases - set(configured_cases))
        if missing:
            raise ValueError(f"requested cases are not in this experiment: {', '.join(missing)}")
        configured_cases = [item for item in configured_cases if item in selected_cases]

    harnesses = _filter_records(list(experiment["harnesses"]), harness_filters, "harness")
    models = _filter_records(list(experiment["models"]), model_filters, "model")
    skills = _filter_records(list(experiment["skill_conditions"]), skill_filters, "skill condition")
    pairs = [
        (harness, model)
        for harness in harnesses
        for model in models
        if _matrix_allows(experiment, str(harness["slug"]), str(model["slug"]))
    ]
    matrix = experiment.get("matrix") or {}
    configured_harnesses = {str(item["slug"]) for item in experiment["harnesses"]}
    configured_models = {str(item["slug"]) for item in experiment["models"]}
    for item in matrix.get("include", []):
        if str(item.get("harness")) not in configured_harnesses:
            raise ValueError(f"matrix references unknown harness: {item.get('harness')}")
        if str(item.get("model")) not in configured_models:
            raise ValueError(f"matrix references unknown model: {item.get('model')}")
    if not pairs:
        raise ValueError("the selected harness/model matrix is empty")
    runs = [
        PlannedRun(case_index[case_id], harness, model, skill, repeat)
        for case_id in configured_cases
        for harness, model in pairs
        for skill in skills
        for repeat in range(int(experiment["repeats"]))
    ]
    return manifest, runs


def create_adapter(harness: dict[str, Any]) -> HarnessAdapter:
    adapter_name = str(harness["adapter"])
    if adapter_name == "mock":
        return MockAdapter()
    if adapter_name == "claude_code":
        return ClaudeCodeAdapter()
    if adapter_name == "codex":
        return CodexAdapter()
    if adapter_name == "external_command":
        return ExternalCommandAdapter()
    raise ValueError(f"adapter {adapter_name!r} is not installed")


def _step_payload(status: str, message: str, elapsed_s: float = 0.0) -> dict[str, Any]:
    return {"status": status, "message": message, "elapsed_s": elapsed_s, "timestamp": utc_now()}


def _real_build_and_run(
    sandbox_path: Path,
    output_path: Path,
    timeout_s: float,
    device_selector: str,
    extra_flags: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bash = resolve_bash()
    if not bash:
        skipped = _step_payload("fail", "bash is required to run the generated wrappers")
        return skipped, _step_payload("skipped", "build did not pass")
    executor = LocalExecutor()
    env = {
        "ONEAPI_DEVICE_SELECTOR": device_selector,
        "EXTRA_SYCL_FLAGS": " ".join(extra_flags),
        "STAGE2_PYTHON": sys.executable,
    }
    build_result = executor.run([bash, "sycl_build.sh"], sandbox_path, timeout_s, env=env)
    build = build_result.as_dict()
    if build_result.status != "pass":
        return build, _step_payload("skipped", "build did not pass")
    sandbox_output = sandbox_path / "output" / "evaluator.txt"
    sandbox_output.parent.mkdir(parents=True, exist_ok=True)
    sandbox_output.unlink(missing_ok=True)
    run_result = executor.run(
        [bash, "sycl_run.sh", "output/evaluator.txt"],
        sandbox_path,
        timeout_s,
        env=env,
    )
    if run_result.status == "pass":
        if sandbox_output.is_file() and not sandbox_output.is_symlink():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sandbox_output, output_path)
        else:
            payload = run_result.as_dict()
            payload["status"] = "fail"
            payload["message"] = "program did not produce output/evaluator.txt"
            return build, payload
    return build, run_result.as_dict()


def _priced_cost(session: dict[str, Any], model: dict[str, Any]) -> float | None:
    if session.get("cost_usd") is not None:
        return float(session["cost_usd"])
    pricing = model.get("pricing_usd_per_million_tokens")
    if not isinstance(pricing, dict):
        return None
    tokens_in = session.get("tokens_in")
    tokens_out = session.get("tokens_out")
    if not isinstance(tokens_in, int) or not isinstance(tokens_out, int):
        return None
    cached_input = session.get("cached_input_tokens")
    if isinstance(cached_input, int) and "cached_input" in pricing:
        cached_input = min(max(cached_input, 0), tokens_in)
        uncached_input = tokens_in - cached_input
        input_cost = (
            uncached_input * float(pricing.get("input", 0))
            + cached_input * float(pricing.get("cached_input", 0))
        )
    else:
        input_cost = tokens_in * float(pricing.get("input", 0))
    return (input_cost + tokens_out * float(pricing.get("output", 0))) / 1_000_000.0


def _finalize_cost(session: dict[str, Any], model: dict[str, Any]) -> None:
    provider_reported = session.get("cost_usd") is not None
    session["cost_usd"] = _priced_cost(session, model)
    if provider_reported:
        session["cost_source"] = session.get("cost_source") or "provider_reported"
        return
    pricing = model.get("pricing_usd_per_million_tokens")
    if session["cost_usd"] is None or not isinstance(pricing, dict):
        session["cost_source"] = session.get("cost_source") or "unavailable"
        return
    session["cost_source"] = "api_price_estimate"
    session["pricing_snapshot"] = {
        "requested_model_id": resolve_model_id(model),
        "rates_usd_per_million_tokens": {
            key: pricing[key]
            for key in ("input", "cached_input", "output")
            if key in pricing
        },
        "source": pricing.get("source"),
        "as_of": pricing.get("as_of"),
    }


def _run_path(
    artifact_root: Path, experiment_id: str, planned: PlannedRun
) -> Path:
    parts = [
        experiment_id,
        safe_component(str(planned.case["case_id"]), "case_id"),
        safe_component(str(planned.harness["slug"]), "harness slug"),
        safe_component(str(planned.model["slug"]), "model slug"),
        safe_component(str(planned.skill_condition["slug"]), "skill condition slug"),
        f"repeat-{planned.repeat}",
    ]
    return artifact_root.joinpath(*parts)


def _remote_config(
    experiment: dict[str, Any], planned: PlannedRun, case_id: str
) -> dict[str, Any] | None:
    executor = experiment["executor"]
    if executor["kind"] != "ssh":
        return None
    extra_flags = [str(item) for item in executor.get("extra_sycl_flags", [])]
    extra_flags.extend(
        str(item)
        for item in executor.get("case_extra_sycl_flags", {}).get(case_id, [])
    )
    return build_remote_config(
        executor,
        (
            str(experiment["experiment_id"]),
            case_id,
            str(planned.harness["slug"]),
            str(planned.model["slug"]),
            str(planned.skill_condition["slug"]),
            f"repeat-{planned.repeat}",
        ),
        extra_flags,
    )


def execute_run(
    experiment: dict[str, Any],
    manifest: dict[str, Any],
    planned: PlannedRun,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    overwrite: bool = False,
) -> tuple[str, Path]:
    experiment_id = safe_component(str(experiment["experiment_id"]), "experiment_id")
    case_id = safe_component(str(planned.case["case_id"]), "case_id")
    harness_slug = safe_component(str(planned.harness["slug"]), "harness slug")
    model_slug = safe_component(str(planned.model["slug"]), "model slug")
    skill_slug = safe_component(str(planned.skill_condition["slug"]), "skill condition slug")
    run_path = _run_path(artifact_root, experiment_id, planned)
    migration_path = run_path / "migration.json"
    if migration_path.exists() and not overwrite:
        return "skipped_existing", migration_path
    if run_path.exists() and overwrite:
        shutil.rmtree(run_path)
    run_path.mkdir(parents=True, exist_ok=True)

    e2e_started = time.perf_counter()
    case_path = (REPO_ROOT / str(planned.case["path"])).resolve()
    include_cuda = experiment.get("reference_mode") == "cuda_output"
    skill_value = planned.skill_condition.get("skill")
    skill_source = resolve_repo_path(str(skill_value)) if skill_value else None
    remote_config = _remote_config(experiment, planned, case_id)
    adapter = create_adapter(planned.harness)
    model_id = resolve_model_id(planned.model, require=not adapter.synthetic)
    session_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"stage2-{case_id}-") as temporary:
        ephemeral_sandbox = create_sandbox(
            case_path,
            Path(temporary) / "workspace",
            include_cuda_tools=include_cuda,
            skill_path=skill_source,
            remote_config=remote_config,
        )
        context = SessionContext(
            case_id=case_id,
            case_path=case_path,
            sandbox_path=ephemeral_sandbox,
            run_path=run_path,
            prompt_path=ephemeral_sandbox / "TASK.md",
            harness=planned.harness,
            model=planned.model,
            model_id=model_id,
            skill_condition=planned.skill_condition,
            skill_path=(ephemeral_sandbox / "skill" / "SKILL.md") if skill_source else None,
            budget=dict(experiment["budget"]),
            executor=LocalExecutor(),
        )
        try:
            session = adapter.run(context).as_dict()
        except Exception as exc:  # An agent failure is a result, not an orchestrator crash.
            session = {
                "status": "error", "tokens_in": None, "tokens_out": None,
                "tokens_total": None, "wall_clock_s": time.perf_counter() - session_started,
                "iterations": 0, "cost_usd": None,
                "cached_input_tokens": None, "reasoning_output_tokens": None,
                "cost_source": None, "pricing_snapshot": None,
                "message": f"{type(exc).__name__}: {exc}",
                "synthetic": bool(adapter.synthetic), "session_id": None,
                "reported_model": None, "duration_api_s": None, "raw_telemetry": None,
            }
        sandbox_path = run_path / "sandbox"
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
        shutil.copytree(ephemeral_sandbox, sandbox_path)
    session["elapsed_harness_s"] = time.perf_counter() - session_started
    _finalize_cost(session, planned.model)
    session["token_budget"] = int(experiment["budget"]["max_tokens"])
    session["token_budget_enforcement"] = "post_run_observation"
    session["token_budget_exceeded"] = bool(
        isinstance(session.get("tokens_total"), int)
        and session["tokens_total"] > session["token_budget"]
    )
    write_json(run_path / "session.json", session)

    # The evaluator, not the agent, owns the build/run contract.
    restore_fixed_wrappers(sandbox_path, include_cuda, remote_config)
    result_output = run_path / "output" / "sycl.txt"
    result_output.parent.mkdir(parents=True, exist_ok=True)
    verify_path = run_path / "verify.json"
    if session.get("synthetic"):
        build = _step_payload("synthetic", "mock adapter does not compile")
        run = _step_payload("synthetic", "known-good output generated by verifier self-test")
        verify = verify_case_output(
            case_path,
            result_output,
            verify_path,
            variant=f"stage2-{harness_slug}-{model_slug}-{skill_slug}-synthetic",
            timeout_s=float(experiment["budget"]["wall_clock_s"]),
            selftest=True,
        )
        funnel = "synthetic"
        migration_success: bool | None = None
        eligible_for_scoring = False
    elif not (sandbox_path / "main.sycl.cpp").is_file() or (sandbox_path / "main.sycl.cpp").is_symlink():
        build = _step_payload("skipped", "adapter did not produce main.sycl.cpp")
        run = _step_payload("skipped", "missing migration source")
        verify = _step_payload("skipped", "missing migration source")
        funnel = "missing"
        migration_success = False
        eligible_for_scoring = True
    else:
        build, run = _real_build_and_run(
            sandbox_path,
            result_output,
            float(experiment["budget"]["wall_clock_s"]),
            str(experiment["executor"]["device_selector"]),
            [str(item) for item in experiment["executor"].get("extra_sycl_flags", [])]
            + [
                str(item)
                for item in experiment["executor"].get("case_extra_sycl_flags", {}).get(case_id, [])
            ],
        )
        if build.get("status") != "pass":
            verify = _step_payload("skipped", "build did not pass")
            funnel = "compile_error"
            migration_success = False
        elif run.get("status") != "pass":
            verify = _step_payload("skipped", "run did not pass")
            funnel = "run_error"
            migration_success = False
        else:
            verify = verify_case_output(
                case_path,
                result_output,
                verify_path,
                variant=f"stage2-{harness_slug}-{model_slug}-{skill_slug}",
                timeout_s=float(experiment["budget"]["wall_clock_s"]),
            )
            migration_success = verify.get("status") == "pass"
            funnel = "pass" if migration_success else "wrong_output"
        eligible_for_scoring = True

    write_json(run_path / "build.json", build)
    write_json(run_path / "run.json", run)
    write_json(verify_path, verify)
    environment = {
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "device_selector": experiment["executor"]["device_selector"],
        "executor_kind": experiment["executor"]["kind"],
        "ssh_target": remote_config.get("target") if remote_config else None,
        "remote_workspace": remote_config.get("remote_workspace") if remote_config else None,
        "oneapi_root": os.environ.get("ONEAPI_ROOT"),
    }
    migration = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "run_id": planned.run_id,
        "repeat": planned.repeat,
        "dataset_id": manifest["dataset_id"],
        "dataset_commit": manifest["dataset_commit"],
        "case_id": case_id,
        "case_path": planned.case["path"],
        "case_fingerprint": planned.case["fingerprint"],
        "category": planned.case["category"],
        "difficulty": planned.case["difficulty"],
        "harness": harness_slug,
        "adapter": planned.harness["adapter"],
        "model": model_slug,
        "model_provider": planned.model.get("provider"),
        "requested_model_id": model_id,
        "reported_model_id": session.get("reported_model"),
        "skill_condition": skill_slug,
        "skill_version": planned.skill_condition.get("version"),
        "prompt_version": experiment["prompt_version"],
        "reference_mode": experiment["reference_mode"],
        "synthetic": bool(session.get("synthetic")),
        "eligible_for_scoring": eligible_for_scoring,
        "funnel": funnel,
        "migration_success": migration_success,
        "e2e_elapsed_s": time.perf_counter() - e2e_started,
        "session": session,
        "build": build,
        "run": run,
        "verify": verify,
        "environment": environment,
        "timestamp": utc_now(),
    }
    write_json(migration_path, migration)
    return funnel, migration_path


def run_experiment(
    experiment_path: Path,
    case_filters: Iterable[str] | None = None,
    harness_filters: Iterable[str] | None = None,
    model_filters: Iterable[str] | None = None,
    skill_filters: Iterable[str] | None = None,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    overwrite: bool = False,
) -> list[tuple[str, Path]]:
    experiment = load_experiment(experiment_path)
    manifest, runs = plan_experiment(
        experiment, case_filters, harness_filters, model_filters, skill_filters
    )
    results: list[tuple[str, Path]] = []
    experiment_id = safe_component(str(experiment["experiment_id"]), "experiment_id")
    for planned in runs:
        try:
            results.append(execute_run(experiment, manifest, planned, artifact_root, overwrite))
        except Exception as exc:  # Keep independent cells running after harness-level faults.
            error_path = _run_path(artifact_root, experiment_id, planned) / "harness_error.json"
            write_json(
                error_path,
                {
                    "schema_version": 2,
                    "experiment_id": experiment_id,
                    "run_id": planned.run_id,
                    "case_id": planned.case["case_id"],
                    "harness": planned.harness["slug"],
                    "model": planned.model["slug"],
                    "skill_condition": planned.skill_condition["slug"],
                    "status": "harness_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "timestamp": utc_now(),
                },
            )
            results.append(("harness_error", error_path))
    return results


def reevaluate_result(
    experiment_path: Path,
    migration_path: Path,
) -> tuple[str, Path]:
    """Repeat only evaluator-owned build/run/verification for an existing session."""
    experiment = load_experiment(experiment_path)
    migration_path = migration_path.resolve()
    migration = read_json(migration_path)
    if migration.get("experiment_id") != experiment.get("experiment_id"):
        raise ValueError("result experiment_id does not match the experiment configuration")
    _, runs = plan_experiment(experiment)
    run_id = str(migration.get("run_id"))
    planned = next((item for item in runs if item.run_id == run_id), None)
    if planned is None:
        raise ValueError(f"run_id is not present in the experiment matrix: {run_id}")

    run_path = migration_path.parent
    sandbox_path = run_path / "sandbox"
    if not (sandbox_path / "main.sycl.cpp").is_file():
        raise ValueError("existing result has no sandbox/main.sycl.cpp to reevaluate")
    case_id = str(planned.case["case_id"])
    case_path = (REPO_ROOT / str(planned.case["path"])).resolve()
    include_cuda = experiment.get("reference_mode") == "cuda_output"
    remote_config = _remote_config(experiment, planned, case_id)
    restore_fixed_wrappers(sandbox_path, include_cuda, remote_config)

    started = time.perf_counter()
    output_path = run_path / "output" / "sycl.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build, run = _real_build_and_run(
        sandbox_path,
        output_path,
        float(experiment["budget"]["wall_clock_s"]),
        str(experiment["executor"]["device_selector"]),
        [str(item) for item in experiment["executor"].get("extra_sycl_flags", [])]
        + [
            str(item)
            for item in experiment["executor"].get("case_extra_sycl_flags", {}).get(case_id, [])
        ],
    )
    verify_path = run_path / "verify.json"
    if build.get("status") != "pass":
        verify = _step_payload("skipped", "build did not pass")
        funnel = "compile_error"
    elif run.get("status") != "pass":
        verify = _step_payload("skipped", "run did not pass")
        funnel = "run_error"
    else:
        verify = verify_case_output(
            case_path,
            output_path,
            verify_path,
            variant=(
                f"stage2-{migration['harness']}-{migration['model']}-"
                f"{migration['skill_condition']}"
            ),
            timeout_s=float(experiment["budget"]["wall_clock_s"]),
        )
        funnel = "pass" if verify.get("status") == "pass" else "wrong_output"
    elapsed = time.perf_counter() - started
    write_json(run_path / "build.json", build)
    write_json(run_path / "run.json", run)
    if verify.get("status") == "skipped":
        write_json(verify_path, verify)
    migration["build"] = build
    migration["run"] = run
    migration["verify"] = verify
    migration["funnel"] = funnel
    migration["migration_success"] = funnel == "pass"
    migration["e2e_elapsed_s"] = float(migration.get("e2e_elapsed_s") or 0) + elapsed
    migration["reevaluation"] = {
        "timestamp": utc_now(),
        "elapsed_s": elapsed,
        "reason": "evaluator_only",
        "model_reinvoked": False,
    }
    write_json(migration_path, migration)
    return funnel, migration_path
