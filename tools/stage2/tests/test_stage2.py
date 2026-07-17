from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from aggregate import aggregate_results  # noqa: E402
from adapters.base import SessionContext  # noqa: E402
from adapters.claude_code import ClaudeCodeAdapter, parse_stream_json  # noqa: E402
from adapters.codex import CodexAdapter, parse_codex_jsonl  # noqa: E402
from adapters.external_command import ExternalCommandAdapter  # noqa: E402
from common import DEFAULT_EXPERIMENT, DEFAULT_MANIFEST, REPO_ROOT, resolve_bash  # noqa: E402
from discovery import audit_frozen_manifest, build_manifest, load_manifest, write_manifest  # noqa: E402
from runner import _finalize_cost, load_experiment, plan_experiment, run_experiment  # noqa: E402
from remote import build_remote_config  # noqa: E402
from sandbox import create_sandbox  # noqa: E402
from verify import verify_case_output  # noqa: E402
from executor import CommandResult  # noqa: E402
from prompts import build_agent_prompt  # noqa: E402


class Stage2HarnessTests(unittest.TestCase):
    def test_frozen_dataset_has_250_ground_truth_cases(self) -> None:
        manifest = build_manifest()
        self.assertEqual(manifest["case_count"], 250)
        self.assertEqual(len({item["case_id"] for item in manifest["cases"]}), 250)

    def test_committed_manifest_and_pilot_are_consistent(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        experiment = load_experiment(DEFAULT_EXPERIMENT)
        loaded, runs = plan_experiment(experiment)
        self.assertEqual(loaded["dataset_id"], manifest["dataset_id"])
        self.assertEqual(len(runs), 10)
        self.assertEqual({run.harness["slug"] for run in runs}, {"mock-harness"})
        self.assertEqual({run.model["slug"] for run in runs}, {"mock-model"})
        self.assertEqual({run.skill_condition["slug"] for run in runs}, {"oob"})

    def test_manifest_count_guard_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            destination = Path(tmp) / "manifest.json"
            with self.assertRaises(ValueError):
                write_manifest(destination, expected_count=249)
            self.assertFalse(destination.exists())

    def test_dataset_audit_exposes_unvalidated_cases_without_mutating_frozen_set(self) -> None:
        audit = audit_frozen_manifest(DEFAULT_MANIFEST)
        self.assertTrue(audit["ready"])
        self.assertEqual(audit["case_count"], 250)
        self.assertEqual(audit["repository_case_count"], 292)
        self.assertEqual(audit["excluded_case_count"], 42)
        self.assertFalse(audit["all_repository_cases_admitted"])

    def test_sandbox_is_whitelisted(self) -> None:
        case_path = REPO_ROOT / "benchmark" / "cases" / "simple-kernels" / "vectorAdd"
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            sandbox = create_sandbox(case_path, Path(tmp) / "sandbox")
            self.assertTrue((sandbox / "main.cu").is_file())
            self.assertTrue((sandbox / "CMakeLists.txt").is_file())
            self.assertTrue((sandbox / "TASK.md").is_file())
            self.assertTrue((sandbox / "sycl_build.sh").is_file())
            self.assertIn("main.sycl.cpp", (sandbox / "sycl_build.sh").read_text(encoding="utf-8"))
            self.assertFalse((sandbox / "metadata.json").exists())
            self.assertFalse((sandbox / "tests").exists())
            self.assertFalse((sandbox / "logs").exists())

    def test_verifier_result_path_does_not_overwrite_stage1_log(self) -> None:
        case_path = REPO_ROOT / "benchmark" / "cases" / "simple-kernels" / "vectorAdd"
        stage1_result = case_path / "logs" / "verify_result.json"
        before = stage1_result.read_bytes()
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            result = verify_case_output(
                case_path,
                root / "output.txt",
                root / "verify.json",
                variant="stage2-test-synthetic",
                selftest=True,
            )
            self.assertEqual(result["status"], "pass")
            self.assertTrue((root / "verify.json").is_file())
        self.assertEqual(stage1_result.read_bytes(), before)

    def test_mock_end_to_end_is_synthetic_and_unscored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            reports = root / "reports"
            results = run_experiment(
                DEFAULT_EXPERIMENT,
                case_filters=["vectorAdd"],
                artifact_root=artifacts,
            )
            self.assertEqual(results[0][0], "synthetic")
            payload = json.loads(results[0][1].read_text(encoding="utf-8"))
            self.assertTrue(payload["synthetic"])
            self.assertFalse(payload["eligible_for_scoring"])
            self.assertIsNone(payload["migration_success"])
            summary = aggregate_results("pilot-v2-offline", artifacts, reports)
            self.assertEqual(summary["total_results"], 1)
            self.assertEqual(summary["scored_results"], 0)
            self.assertEqual(summary["synthetic_results"], 1)

    def test_one_cell_failure_does_not_stop_later_cells(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            experiment = json.loads(DEFAULT_EXPERIMENT.read_text(encoding="utf-8"))
            experiment["experiment_id"] = "failure-isolation-test"
            experiment["case_ids"] = ["vectorAdd", "reduceSum"]
            experiment["harnesses"] = [{"slug": "missing-harness", "adapter": "not-installed"}]
            experiment["matrix"] = {
                "include": [{"harness": "missing-harness", "model": "mock-model"}]
            }
            experiment_path = root / "experiment.json"
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            results = run_experiment(experiment_path, artifact_root=root / "artifacts")
            self.assertEqual(len(results), 2)
            self.assertEqual([status for status, _ in results], ["harness_error", "harness_error"])
            self.assertTrue(all(path.is_file() for _, path in results))

    def test_skill_is_present_only_in_with_skill_condition(self) -> None:
        case_path = REPO_ROOT / "benchmark" / "cases" / "simple-kernels" / "vectorAdd"
        skill = REPO_ROOT / "benchmark" / "stage2" / "skills" / "cuda-to-sycl-migration"
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            oob = create_sandbox(case_path, root / "oob")
            with_skill = create_sandbox(case_path, root / "with", skill_path=skill)
            self.assertFalse((oob / "skill").exists())
            self.assertTrue((with_skill / "skill" / "SKILL.md").is_file())
            self.assertTrue((with_skill / "skill" / "references" / "patterns.md").is_file())

    def test_canonical_prompt_differs_only_by_skill_condition(self) -> None:
        oob = build_agent_prompt(False)
        with_skill = build_agent_prompt(True)
        self.assertIn("No migration skill is supplied", oob)
        self.assertNotIn("skill/SKILL.md", oob)
        self.assertIn("skill/SKILL.md", with_skill)
        self.assertTrue(oob.endswith("wrappers pass."))
        self.assertTrue(with_skill.endswith("wrappers pass."))

    def test_remote_sandbox_contains_only_model_visible_proxy_configuration(self) -> None:
        case_path = REPO_ROOT / "benchmark" / "cases" / "simple-kernels" / "vectorAdd"
        config = build_remote_config(
            {
                "kind": "ssh",
                "target": "ubuntu@intel-worker",
                "remote_root": "/tmp/stage2",
                "device_selector": "level_zero:gpu",
            },
            ("experiment", "vectorAdd", "harness", "model", "oob", "repeat-0"),
            [],
        )
        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            sandbox = create_sandbox(
                case_path, Path(tmp) / "sandbox", remote_config=config
            )
            self.assertTrue((sandbox / "remote_exec.py").is_file())
            self.assertTrue((sandbox / "remote_config.json").is_file())
            self.assertIn("remote_exec.py build", (sandbox / "sycl_build.sh").read_text())
            self.assertIn("remote_exec.py run", (sandbox / "sycl_run.sh").read_text())
            remote_text = (sandbox / "remote_config.json").read_text(encoding="utf-8")
            self.assertIn("ubuntu@intel-worker", remote_text)
            self.assertNotIn("password", remote_text.lower())
            self.assertFalse((sandbox / "tests").exists())

    @unittest.skipUnless(os.name == "nt", "Windows-specific bash selection")
    def test_windows_prefers_git_bash_over_wsl_bash(self) -> None:
        bash = resolve_bash()
        self.assertIsNotNone(bash)
        self.assertIn("Git", str(bash))

    def test_claude_stream_json_telemetry_is_normalized(self) -> None:
        stream = "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 100, "output_tokens": 20}}}),
                json.dumps(
                    {
                        "type": "result", "is_error": False, "session_id": "session-1",
                        "num_turns": 2, "duration_ms": 1234, "duration_api_ms": 1000,
                        "total_cost_usd": 0.12, "model": "exact-opus-id", "result": "done",
                    }
                ),
            ]
        )
        parsed = parse_stream_json(stream)
        self.assertEqual(parsed["status"], "completed")
        self.assertEqual(parsed["tokens_total"], 120)
        self.assertEqual(parsed["iterations"], 2)
        self.assertEqual(parsed["session_id"], "session-1")

    def test_baseline_plan_has_harness_model_and_two_conditions(self) -> None:
        baseline = REPO_ROOT / "benchmark" / "stage2" / "experiments" / "baseline_claude_opus.json"
        experiment = load_experiment(baseline)
        self.assertEqual(experiment["executor"]["kind"], "ssh")
        _, runs = plan_experiment(experiment, case_filters=["vectorAdd"])
        self.assertEqual(len(runs), 2)
        self.assertEqual({run.harness["slug"] for run in runs}, {"claude-code"})
        self.assertEqual({run.model["slug"] for run in runs}, {"opus-4x"})
        self.assertEqual({run.skill_condition["slug"] for run in runs}, {"oob", "with-sycl-skill"})

    def test_codex_baseline_is_pinned_and_has_two_conditions(self) -> None:
        baseline = (
            REPO_ROOT / "benchmark" / "stage2" / "experiments"
            / "baseline_codex_54mini.json"
        )
        experiment = load_experiment(baseline)
        _, runs = plan_experiment(experiment, case_filters=["vectorAdd"])
        self.assertEqual(len(runs), 2)
        self.assertEqual({run.harness["adapter"] for run in runs}, {"codex"})
        self.assertEqual({run.model["model_id"] for run in runs}, {"gpt-5.4-mini"})
        self.assertFalse(runs[0].model["require_immutable_model_id"])
        self.assertEqual({run.skill_condition["slug"] for run in runs}, {"oob", "with-sycl-skill"})

    def test_codex_jsonl_and_cached_price_are_normalized(self) -> None:
        stream = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "error", "message": "transient reconnect"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "command_execution", "command": "build"},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "done"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10000,
                            "cached_input_tokens": 4000,
                            "output_tokens": 1000,
                            "reasoning_output_tokens": 600,
                        },
                    }
                ),
            ]
        )
        parsed = parse_codex_jsonl(stream)
        self.assertEqual(parsed["status"], "completed")
        self.assertEqual(parsed["tokens_total"], 11000)
        self.assertEqual(parsed["cached_input_tokens"], 4000)
        self.assertEqual(parsed["reasoning_output_tokens"], 600)
        self.assertEqual(parsed["iterations"], 1)
        session = {
            "tokens_in": parsed["tokens_in"],
            "tokens_out": parsed["tokens_out"],
            "cached_input_tokens": parsed["cached_input_tokens"],
            "cost_usd": None,
        }
        model = {
            "model_id": "fixed-model",
            "pricing_usd_per_million_tokens": {
                "input": 1.0,
                "cached_input": 0.1,
                "output": 5.0,
                "source": "official",
                "as_of": "2026-01-01",
            },
        }
        _finalize_cost(session, model)
        self.assertAlmostEqual(session["cost_usd"], 0.0114)
        self.assertEqual(session["cost_source"], "api_price_estimate")
        self.assertEqual(session["pricing_snapshot"]["requested_model_id"], "fixed-model")

    def test_claude_adapter_builds_headless_command_and_saves_raw_logs(self) -> None:
        class FakeExecutor:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            def run(self, argv, cwd, timeout_s, env=None):
                command = [str(item) for item in argv]
                self.commands.append(command)
                if "--version" in command:
                    stdout = "1.2.3\n"
                else:
                    (Path(cwd) / "main.sycl.cpp").write_text("int main(){return 0;}", encoding="utf-8")
                    stdout = "\n".join(
                        [
                            json.dumps({"type": "assistant", "message": {"model": "exact-opus", "usage": {"input_tokens": 7, "output_tokens": 3}}}),
                            json.dumps({"type": "result", "is_error": False, "session_id": "s", "num_turns": 1, "total_cost_usd": 0.01, "result": "done"}),
                        ]
                    )
                return CommandResult(command, str(cwd), "pass", 0, 0.01, stdout, "", "now")

        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            (sandbox / "TASK.md").write_text("task", encoding="utf-8")
            fake = FakeExecutor()
            context = SessionContext(
                case_id="vectorAdd", case_path=root, sandbox_path=sandbox, run_path=root,
                prompt_path=sandbox / "TASK.md",
                harness={"command": "claude", "allowed_tools": ["Read", "Write"]},
                model={"slug": "opus"}, model_id="exact-opus",
                skill_condition={"slug": "oob"}, skill_path=None,
                budget={"max_iterations": 5, "max_tokens": 1000, "wall_clock_s": 60},
                executor=fake,
            )
            with patch("adapters.claude_code.shutil.which", return_value="/fake/claude"):
                result = ClaudeCodeAdapter().run(context)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tokens_total, 10)
            self.assertEqual(result.reported_model, "exact-opus")
            self.assertTrue((root / "harness_stdout.jsonl").is_file())
            self.assertIn("--output-format", fake.commands[1])
            self.assertIn("stream-json", fake.commands[1])
            self.assertIn("--allowedTools", fake.commands[1])
            prompt_index = fake.commands[1].index("-p") + 1
            self.assertEqual(fake.commands[1][prompt_index], build_agent_prompt(False))
            self.assertEqual(
                (root / "agent_prompt.txt").read_text(encoding="utf-8").strip(),
                build_agent_prompt(False),
            )
            self.assertTrue((sandbox / "AGENT_PROMPT.md").is_file())

    def test_codex_adapter_builds_isolated_headless_command_and_saves_usage(self) -> None:
        class FakeExecutor:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []

            def run(self, argv, cwd, timeout_s, env=None):
                command = [str(item) for item in argv]
                self.commands.append(command)
                (Path(cwd) / "main.sycl.cpp").write_text(
                    "int main(){return 0;}", encoding="utf-8"
                )
                stdout = "\n".join(
                    [
                        json.dumps({"type": "thread.started", "thread_id": "codex-session"}),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 7,
                                    "cached_input_tokens": 2,
                                    "output_tokens": 3,
                                    "reasoning_output_tokens": 1,
                                },
                            }
                        ),
                    ]
                )
                return CommandResult(command, str(cwd), "pass", 0, 0.01, stdout, "", "now")

        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            fake = FakeExecutor()
            context = SessionContext(
                case_id="vectorAdd", case_path=root, sandbox_path=sandbox, run_path=root,
                prompt_path=sandbox / "TASK.md",
                harness={
                    "command": "codex",
                    "reasoning_effort": "low",
                    "externally_isolated_bypass": True,
                },
                model={"slug": "mini"}, model_id="fixed-mini",
                skill_condition={"slug": "oob"}, skill_path=None,
                budget={"max_iterations": 5, "max_tokens": 1000, "wall_clock_s": 60},
                executor=fake,
            )
            with patch(
                "adapters.codex.resolve_codex_cli",
                return_value=("/fake/codex", "codex-cli 1.2.3"),
            ):
                result = CodexAdapter().run(context)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tokens_total, 10)
            self.assertEqual(result.cached_input_tokens, 2)
            self.assertEqual(result.reasoning_output_tokens, 1)
            self.assertTrue((root / "harness_stdout.jsonl").is_file())
            command = fake.commands[0]
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--ignore-rules", command)
            self.assertIn("remote_plugin", command)
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
            self.assertIn("fixed-mini", command)
            self.assertEqual(command[-1], build_agent_prompt(False))
            self.assertTrue((sandbox / "AGENT_PROMPT.md").is_file())

    def test_external_adapter_receives_canonical_prompt_and_placeholders(self) -> None:
        class FakeExecutor:
            def __init__(self) -> None:
                self.command: list[str] = []

            def run(self, argv, cwd, timeout_s, env=None):
                self.command = [str(item) for item in argv]
                Path(cwd, "main.sycl.cpp").write_text(
                    "int main(){return 0;}", encoding="utf-8"
                )
                Path(cwd, "stage2_telemetry.json").write_text(
                    json.dumps(
                        {
                            "tokens_in": 11,
                            "tokens_out": 5,
                            "iterations": 2,
                            "model": "reported-exact-model",
                        }
                    ),
                    encoding="utf-8",
                )
                return CommandResult(
                    self.command, str(cwd), "pass", 0, 0.01, "", "", "now"
                )

        with tempfile.TemporaryDirectory(prefix="stage2-test-") as tmp:
            root = Path(tmp)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            (sandbox / "TASK.md").write_text("task", encoding="utf-8")
            fake = FakeExecutor()
            context = SessionContext(
                case_id="vectorAdd", case_path=root, sandbox_path=sandbox, run_path=root,
                prompt_path=sandbox / "TASK.md",
                harness={
                    "argv": [
                        "fake-harness", "--model", "{model_id}", "--workspace", "{sandbox}",
                        "--prompt", "{prompt}", "--prompt-file", "{agent_prompt_file}",
                        "--skill", "{skill_file}",
                    ],
                    "telemetry_file": "stage2_telemetry.json",
                },
                model={"slug": "model"}, model_id="exact-model",
                skill_condition={"slug": "oob"}, skill_path=None,
                budget={"max_iterations": 5, "max_tokens": 1000, "wall_clock_s": 60},
                executor=fake,
            )
            with patch(
                "adapters.external_command.shutil.which", return_value="/fake/harness"
            ):
                result = ExternalCommandAdapter().run(context)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tokens_total, 16)
            self.assertEqual(result.reported_model, "reported-exact-model")
            self.assertIn(build_agent_prompt(False), fake.command)
            self.assertIn(str(sandbox / "AGENT_PROMPT.md"), fake.command)
            self.assertEqual(fake.command[-1], "")


if __name__ == "__main__":
    unittest.main()
