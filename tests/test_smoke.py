"""Offline S3 tests. All training/identity/memory data is synthetic."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from fitcheck import FSDPConfig, LoRAConfig, TrainingSpec
from fitcheck.smoke import SmokeLimits, SmokePlan, json_text, render_slurm, write_smoke_job
from fitcheck.smoke_report import GPUPeak, RunState, build_report, measured_metrics, summary
from fitcheck.smoke_runner import run_steps, supervise
from smoke_fakes import FakeRunner


ROOT = Path(__file__).resolve().parents[1]


def spec(**changes):
    return TrainingSpec(**(dict(model_id="Qwen/Qwen3-4B", model_revision="test-revision",
                               training_method="full", sequence_length=2048,
                               micro_batch_size=4, max_steps=1000,
                               gradient_accumulation_steps=2) | changes))


def completed(plan):
    return run_steps(plan, FakeRunner)


def predictions(plan, state):
    return {"source": "synthetic test fixture, not calibrated", "spec": plan.original.to_dict(),
            "spec_fingerprint": plan.original.fingerprint(), "identity": deepcopy(state.identity),
            "metrics": {key: {"definition": value["definition"], "estimate": value["value"],
                               "upper": value["value"] * 1.1, "duration_invariant": True}
                        for key, value in measured_metrics(state, plan.smoke.gpu_count).items()}}


class PlanningTests(unittest.TestCase):
    def test_preserves_every_non_duration_field_and_fingerprints(self):
        configs = [spec(), spec(training_method="lora", lora=LoRAConfig(16, 32, ("all-linear",))),
                   spec(training_method="qlora", lora=LoRAConfig(16, 32, ("all-linear",)), quantization="nf4"),
                   spec(gpu_count=4, distributed_strategy="fsdp",
                        fsdp=FSDPConfig("full_shard", ("Qwen3DecoderLayer",), True))]
        for original in configs:
            with self.subTest(original=original):
                plan = SmokePlan(original)
                for key, value in original.to_dict().items():
                    if key not in ("max_steps", "num_epochs"):
                        self.assertEqual(plan.smoke.to_dict()[key], value)
                self.assertEqual(plan.smoke.max_steps, 13)
                self.assertNotEqual(original.fingerprint(), plan.smoke.fingerprint())
                self.assertEqual(plan.to_dict()["overrides"], {"max_steps": {"original": 1000, "smoke": 13}})
                self.assertEqual(SmokePlan.from_dict(json.loads(json_text(plan.to_dict()))), plan)

    def test_never_extends_short_jobs_and_epochs_require_external_accounting(self):
        plan = SmokePlan(spec(max_steps=1))
        self.assertEqual((plan.warmup_steps, plan.measured_steps), (0, 1))
        self.assertEqual(plan.to_dict()["overrides"], {})
        self.assertEqual(plan.original.fingerprint(), plan.smoke.fingerprint())
        epoch_spec = spec(max_steps=None, num_epochs=0.5)
        with self.assertRaises(ValueError):
            SmokePlan(epoch_spec)
        plan = SmokePlan(epoch_spec, epoch_step_count=2)
        self.assertEqual((plan.warmup_steps, plan.measured_steps, plan.smoke.max_steps), (1, 1, 2))
        self.assertEqual(set(plan.to_dict()["overrides"]), {"max_steps", "num_epochs"})
        with self.assertRaises(ValueError):
            SmokePlan(spec(), epoch_step_count=200)

    def test_invalid_limits_and_tampered_plan(self):
        for changes in (dict(wall_seconds=1), dict(wall_seconds=86401), dict(warmup_steps=-1),
                        dict(measured_steps=0), dict(measured_steps=True), dict(wall_seconds=3.5)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SmokeLimits(**changes)
        original = SmokePlan(spec()).to_dict()
        for key, value in (("schema_version", 2), ("smoke_fingerprint", "wrong"), ("warmup_steps", 0)):
            with self.assertRaises(ValueError):
                SmokePlan.from_dict(original | {key: value})
        data = deepcopy(original)
        data["smoke_spec"]["gpu_count"] = 4
        with self.assertRaises(ValueError):
            SmokePlan.from_dict(data)

    def test_script_bundle_syntax_quoting_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="fitcheck space '") as directory:
            root = Path(directory)
            full = root / "train.sbatch"
            full.write_text("existing full job")
            plan = SmokePlan(spec(gpu_count=4, distributed_strategy="fsdp",
                                  fsdp=FSDPConfig("full_shard", ("Layer",), True)))
            options = dict(adapter="harness.runner:create", project_dir=str(root / "project $(nope)"),
                           account="project", partition="boost_usr_prod", cpus_per_task=8)
            path = write_smoke_job(root, plan, **options)
            text = path.read_text()
            self.assertIn("#SBATCH --time=00:05:00", text)
            self.assertIn("#SBATCH --gres=gpu:4", text)
            self.assertIn("HF_HUB_OFFLINE=1", text)
            self.assertIn("UV_PYTHON_DOWNLOADS=never", text)
            self.assertIn("uv run --offline --no-sync", text)
            self.assertNotIn("sbatch ", text)
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(full.read_text(), "existing full job")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_smoke_job(root, plan, **options)
            self.assertEqual(path.read_bytes(), before)
            request = json.loads((path.parent / "request.json").read_text())
            self.assertEqual(SmokePlan.from_dict(request["plan"]), plan)
            self.assertIsNone(request["predictions"])

    def test_script_rejects_directive_injection(self):
        options = dict(adapter="harness:factory", project_dir="/work/project",
                       request_path="/work/job/request.json", account="project",
                       partition="gpu", cpus_per_task=4)
        for changes in (dict(account="p\n#SBATCH --nodes=10"), dict(partition="gpu;echo nope"),
                        dict(adapter="module:factory;echo nope"), dict(project_dir="relative"),
                        dict(request_path="/work\n/req"), dict(cpus_per_task=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                render_slurm(SmokePlan(spec()), **(options | changes))


class MeasurementTests(unittest.TestCase):
    def test_real_update_contract_warmup_exclusion_and_separate_gpus(self):
        plan = SmokePlan(spec(gpu_count=4, distributed_strategy="fsdp",
                              fsdp=FSDPConfig("full_shard", ("Layer",), True)), SmokeLimits(measured_steps=3))
        runner = FakeRunner(plan.original, plan.smoke)
        state = run_steps(plan, lambda original, smoke: runner)
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.step_seconds, [1.0, 2.0, 3.0])
        self.assertEqual(runner.events[:4], ["step", "step", "step", "reset"])
        self.assertTrue(runner.closed)
        report = build_report(plan, state)
        self.assertEqual(report["metrics"]["optimizer_step_seconds"]["value"], 2.0)
        self.assertEqual(report["metrics"]["gpu/0/reserved_bytes"]["value"], 203)
        self.assertEqual(report["metrics"]["gpu/3/reserved_bytes"]["value"], 206)
        device = report["metrics"]["gpu/0/device_sampled_bytes"]
        self.assertEqual(device["definition"]["unit"], "bytes")
        self.assertTrue(device["definition"]["sampled"])
        self.assertEqual(device["definition"]["sample_interval_seconds"], 0.1)
        self.assertEqual(json.loads(json_text(report)), report)
        self.assertIn("does not guarantee", summary(report))

    def test_failure_statuses_preserve_partial_measurements(self):
        plan = SmokePlan(spec(), SmokeLimits(warmup_steps=1, measured_steps=4))
        for failure, expected in (("oom", "oom"), ("error", "error"), ("short", "incomplete"),
                                  ("load_only", "error")):
            with self.subTest(failure=failure):
                runner = FakeRunner(plan.original, plan.smoke, failure=failure)
                state = run_steps(plan, lambda original, smoke: runner)
                self.assertEqual(state.status, expected)
                self.assertTrue(runner.closed)
                if failure != "load_only":
                    self.assertEqual(len(state.step_seconds), 1)
                self.assertTrue(all(c["status"] == "unavailable" for c in
                                    build_report(plan, state)["comparisons"].values()))

    def test_wrong_applied_spec_and_missing_rank_metrics_fail(self):
        plan = SmokePlan(spec())
        runner = FakeRunner(plan.original, replace(plan.smoke, micro_batch_size=1))
        self.assertEqual(run_steps(plan, lambda a, b: runner).status, "error")
        runner = FakeRunner(plan.original, plan.smoke)
        runner.peak_memory = lambda: []
        self.assertEqual(run_steps(plan, lambda a, b: runner).status, "error")

    def test_invalid_memory_and_sampling_metadata(self):
        for values in (dict(allocated_bytes=-1), dict(allocated_bytes=True),
                       dict(allocated_bytes=200, reserved_bytes=100),
                       dict(device_sampled_bytes=100), dict(sample_count=1),
                       dict(device_sampled_bytes=100, sample_interval_seconds=float("nan"), sample_count=1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GPUPeak(0, **values)

    def test_device_samples_may_start_after_the_first_measured_update(self):
        plan = SmokePlan(spec(), SmokeLimits(warmup_steps=0, measured_steps=2))
        runner = FakeRunner(plan.original, plan.smoke)
        collect = runner.peak_memory
        runner.peak_memory = lambda: [GPUPeak(0, 101, 201)] if runner.steps == 1 else collect()
        state = run_steps(plan, lambda a, b: runner)
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.gpu_peaks[0].sample_count, 2)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.plan = SmokePlan(spec(), SmokeLimits(measured_steps=3))
        self.state = completed(self.plan)
        self.predictions = predictions(self.plan, self.state)

    def compare(self, prediction=None, state=None):
        return build_report(self.plan, state or self.state,
                            self.predictions if prediction is None else prediction)

    def test_comparable_and_exceeded_bounds(self):
        report = self.compare()
        self.assertTrue(all(c["status"] == "comparable" for c in report["comparisons"].values()))
        self.predictions["metrics"]["optimizer_step_seconds"].update(estimate=1, upper=1.5)
        report = self.compare()
        self.assertTrue(report["comparisons"]["optimizer_step_seconds"]["exceeds_upper_bound"])
        self.assertTrue(any("exceeds" in warning for warning in report["warnings"]))

    def test_mismatched_workload_environment_and_unknown_identity(self):
        for kind in ("workload", "hardware", "software", "revision", "dataset", "runtime", "missing"):
            prediction = deepcopy(self.predictions)
            if kind == "workload":
                changed = replace(self.plan.original, micro_batch_size=1)
                prediction.update(spec=changed.to_dict(), spec_fingerprint=changed.fingerprint())
            elif kind == "hardware":
                prediction["identity"]["hardware"]["gpus"][0]["name"] = "Other GPU"
            elif kind == "software":
                prediction["identity"]["software"]["torch"] = "different"
            elif kind == "missing":
                prediction["identity"] = None
            else:
                key = {"revision": "resolved_model_revision", "dataset": "dataset_fingerprint",
                       "runtime": "execution_fingerprint"}[kind]
                prediction["identity"][key] = "different"
            with self.subTest(kind=kind):
                report = self.compare(prediction)
                self.assertTrue(all(c["status"] == "unavailable" for c in report["comparisons"].values()))
                self.assertTrue(all(c["reasons"] for c in report["comparisons"].values()))
        self.state.identity = self.predictions["identity"] = {}
        self.assertEqual(self.compare()["comparisons"]["optimizer_step_seconds"]["status"], "unavailable")

    def test_metric_definition_unit_sampling_and_duration_mismatches(self):
        key = "gpu/0/device_sampled_bytes"
        for field, value in (("unit", "GiB"), ("sampled", False), ("sample_interval_seconds", 1.0),
                             ("source", "torch.cuda.max_memory_reserved"), ("scope", "all_gpus"),
                             ("window", "includes_warmup")):
            prediction = deepcopy(self.predictions)
            prediction["metrics"][key]["definition"][field] = value
            with self.subTest(field=field):
                self.assertEqual(self.compare(prediction)["comparisons"][key]["status"], "unavailable")
        self.predictions["metrics"]["optimizer_step_seconds"].pop("duration_invariant")
        self.assertIn("duration", " ".join(self.compare()["comparisons"]["optimizer_step_seconds"]["reasons"]))

    def test_missing_predictions_measurements_and_bounds_are_unavailable(self):
        report = build_report(self.plan, self.state)
        comparison = report["comparisons"]["optimizer_step_seconds"]
        self.assertIsNone(report["predictions"])
        self.assertIsNone(comparison["prediction"])
        self.assertIsNone(comparison["exceeds_upper_bound"])
        self.state.gpu_peaks = [GPUPeak(0)]
        self.assertIn("measurement unavailable", self.compare()["comparisons"]["gpu/0/reserved_bytes"]["reasons"])
        self.predictions["metrics"]["optimizer_step_seconds"].pop("upper")
        self.assertIsNone(self.compare()["comparisons"]["optimizer_step_seconds"]["exceeds_upper_bound"])

    def test_total_duration_partial_runs_and_bad_predictions_never_compare(self):
        self.predictions["metrics"]["full_training_seconds"] = {"estimate": 1000, "upper": 2000}
        report = self.compare()
        self.assertEqual(report["comparisons"]["full_training_seconds"]["status"], "unavailable")
        self.state.status = "incomplete"
        self.assertEqual(self.compare()["comparisons"]["optimizer_step_seconds"]["status"], "unavailable")
        self.state.status = "completed"
        self.state.step_seconds.pop()
        self.assertEqual(self.compare()["comparisons"]["optimizer_step_seconds"]["status"], "unavailable")
        self.assertEqual(self.compare({})["comparisons"]["optimizer_step_seconds"]["status"], "unavailable")


class SupervisorTests(unittest.TestCase):
    def run_mock(self, root, name, limits=SmokeLimits(warmup_steps=1, measured_steps=3)):
        plan = SmokePlan(spec(), limits)
        request = root / "request.json"
        request.write_text(json_text({"plan": plan.to_dict(), "predictions": None}))
        with patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "tests")))}):
            return supervise(request, "smoke_fakes:" + name, root / "output")

    def test_supervised_success_oom_crash_and_output_no_overwrite(self):
        for name, expected in (("ok", "completed"), ("oom", "oom"), ("crash", "error"), ("missing_factory", "error")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = self.run_mock(root, name)
                self.assertEqual(report["execution"]["status"], expected)
                self.assertEqual(json.loads((root / "output/report.json").read_text()), report)
                self.assertIn(expected, (root / "output/summary.txt").read_text())
                with self.assertRaises(FileExistsError):
                    supervise(root / "request.json", "smoke_fakes:ok", root / "output")

    def test_deadline_covers_initialization_steps_and_cleanup(self):
        for name in ("init_hang", "hang", "close_hang"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                report = self.run_mock(Path(directory), name,
                                       SmokeLimits(wall_seconds=2, warmup_steps=0, measured_steps=1))
                self.assertEqual(report["execution"]["status"], "incomplete")
                self.assertIn("deadline", report["execution"]["detail"])
                self.assertLess(report["execution"]["elapsed_seconds"], 5)

    def test_sigterm_keeps_a_report_and_cli_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, output = root / "request.json", root / "output"
            request.write_text(json_text({"plan": SmokePlan(spec()).to_dict(), "predictions": None}))
            environment = os.environ | {"PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "tests")))}
            process = subprocess.Popen(
                [sys.executable, "-S", "-m", "fitcheck.smoke_runner", "--request", str(request),
                 "--adapter", "smoke_fakes:hang", "--output", str(output)],
                cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    checkpoint = output / "checkpoint.json"
                    if checkpoint.exists() and json.loads(checkpoint.read_text()).get("identity"):
                        break
                    time.sleep(0.01)
                else:
                    self.fail("mock worker did not initialize")
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 1, stderr)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["execution"]["status"], "incomplete")
                self.assertIn("signal", report["execution"]["detail"])
                self.assertIn("incomplete", stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
