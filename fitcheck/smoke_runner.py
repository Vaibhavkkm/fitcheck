"""Bounded orchestration of an external shared training runner (no GPU imports)."""

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Protocol

from .smoke import SmokePlan, json_text, validate_adapter
from .smoke_report import GPUPeak, RunState, StepResult, TrainingOOM, build_report, summary
from .training_spec import TrainingSpec


class TrainingRunner(Protocol):
    """Adapter owns model/data loading, distributed launch and all training math."""

    spec: TrainingSpec  # Actual applied specification, not just requested settings.

    def identity(self) -> dict: ...
    def train_step(self) -> StepResult: ...
    def reset_measurements(self) -> None: ...
    def peak_memory(self) -> list[GPUPeak]: ...
    def close(self) -> None: ...


RunnerFactory = Callable[[TrainingSpec, TrainingSpec], TrainingRunner]


def run_steps(plan: SmokePlan, factory: RunnerFactory, *,
              publish: Callable[[RunState], None] = lambda state: None,
              clock: Callable[[], float] = time.monotonic) -> RunState:
    """Cooperative update cap; the CLI supervisor also enforces a hard deadline."""
    state, runner = RunState(), None
    started = clock()

    def checkpoint():
        state.elapsed_seconds = max(0.0, clock() - started)
        publish(state)

    def update():
        if clock() - started >= plan.limits.execution_seconds:
            raise TimeoutError("smoke execution budget exhausted")
        step = runner.train_step()
        if not isinstance(step, StepResult):
            raise ValueError("train_step must return StepResult")
        step.validate(plan.smoke)
        return step

    try:
        checkpoint()
        runner = factory(plan.original, plan.smoke)
        state.applied_spec_fingerprint = runner.spec.fingerprint()
        if state.applied_spec_fingerprint != plan.smoke.fingerprint():
            raise ValueError("runner applied a different TrainingSpec")
        state.identity = json.loads(json_text(runner.identity()))
        checkpoint()
        for _ in range(plan.warmup_steps):
            update()
            state.warmup_completed += 1
            checkpoint()
        # The adapter synchronizes ALL ranks before resetting allocator peaks
        # and restarting the device-memory sampler. Warm-up peaks are excluded.
        runner.reset_measurements()
        for _ in range(plan.measured_steps):
            step = update()
            state.step_seconds.append(step.seconds)
            peaks = runner.peak_memory()
            if (not isinstance(peaks, list) or len(peaks) != plan.smoke.gpu_count
                    or any(not isinstance(p, GPUPeak) for p in peaks)
                    or sorted(p.rank for p in peaks) != list(range(plan.smoke.gpu_count))):
                raise ValueError("peak_memory must return exactly one GPUPeak per GPU rank")
            peaks.sort(key=lambda p: p.rank)
            for previous, current in zip(state.gpu_peaks, peaks):
                for field in ("allocated_bytes", "reserved_bytes", "device_sampled_bytes", "sample_count"):
                    old, new = getattr(previous, field), getattr(current, field)
                    if old is not None and (new is None or new < old):
                        raise ValueError("peak memory/sample counts decreased inside the measurement window")
                if previous.sample_count and previous.sample_interval_seconds != current.sample_interval_seconds:
                    raise ValueError("device sampling interval changed inside the measurement window")
            state.gpu_peaks = peaks
            checkpoint()
        state.status, state.detail = "completed", "All bounded warm-up and measured optimizer updates completed."
    except TrainingOOM as exc:
        state.status, state.detail = "oom", str(exc) or "Runner reported device OOM."
    except (StopIteration, TimeoutError) as exc:
        state.status, state.detail = "incomplete", str(exc) or "Runner exhausted data before the requested updates."
    except Exception as exc:
        state.status, state.detail = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if state.status != "completed":
            checkpoint()  # Preserve confirmed OOM/errors even if cleanup hangs.
        if runner is not None:
            try:
                runner.close()
            except Exception as exc:
                if state.status == "completed":
                    state.status = "error"
                state.detail += f" Cleanup failed: {type(exc).__name__}: {exc}"
        checkpoint()
    return state


def _atomic_json(path: Path, data: dict):
    # Only used within a newly claimed output directory owned by this invocation.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json_text(data), encoding="utf-8")
    temporary.replace(path)


def _read_request(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    json_text(data)  # Reject NaN/Infinity even though Python's decoder accepts them.
    if not isinstance(data, dict) or set(data) != {"plan", "predictions"}:
        raise ValueError("request must contain exactly plan and predictions")
    return SmokePlan.from_dict(data["plan"]), data["predictions"]


def _worker(request: Path, adapter: str, output: Path) -> int:
    plan, _ = _read_request(request)
    def factory(original, smoke):
        module, name = adapter.split(":")
        return getattr(importlib.import_module(module), name)(original, smoke)
    state = run_steps(plan, factory, publish=lambda state:
                      _atomic_json(output / "checkpoint.json", state.to_dict()))
    return 0 if state.status == "completed" else 1


def supervise(request: Path, adapter: str, output: Path) -> dict:
    """POSIX supervisor: includes import, model initialization, steps and cleanup.

    The worker and its local distributed children share a process group. Adapters
    must not detach children or submit jobs. Slurm remains the outer hard limit.
    """
    validate_adapter(adapter)
    plan, predictions = _read_request(request)
    # Validate reporting input before starting any runner.
    build_report(plan, RunState(), predictions)
    output = Path(output)
    output.mkdir(mode=0o700)  # Never overwrite a previous run, even an empty one.
    _atomic_json(output / "checkpoint.json", RunState().to_dict())
    # Save a recoverable incomplete report before spawning the worker.
    initial = build_report(plan, RunState(), predictions)
    _atomic_json(output / "report.json", initial)
    (output / "summary.txt").write_text(summary(initial), encoding="utf-8")
    interrupted = []
    handlers = {}
    process = None
    started = time.monotonic()
    reason = None
    returncode = None
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, lambda number, frame: interrupted.append(number))
        with (output / "runner.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, *(["-S"] if sys.flags.no_site else []),
                 "-m", "fitcheck.smoke_runner", "--worker",
                 "--request", str(Path(request).resolve()), "--adapter", adapter,
                 "--output", str(output.resolve())],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while process.poll() is None:
                if interrupted:
                    reason = f"interrupted by signal {interrupted[0]}"
                    break
                remaining = plan.limits.execution_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    reason = "execution deadline exceeded (including runner initialization/cleanup)"
                    break
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            if interrupted and reason is None:
                reason = f"interrupted by signal {interrupted[0]}"
    except Exception as exc:
        reason = f"supervisor error: {type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            # Also stop any remaining local rank/sampler children after an exit.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=min(2, max(0.1, plan.limits.wall_seconds - (time.monotonic() - started))))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                reason = "worker did not exit after SIGKILL; Slurm must enforce the allocation limit"
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    try:
        state = RunState.from_dict(json.loads((output / "checkpoint.json").read_text(encoding="utf-8")))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        state = RunState(status="error", detail=f"Cannot recover worker checkpoint: {exc}")
    state.elapsed_seconds = time.monotonic() - started
    if reason:
        # A deadline/signal cannot be called a completed run, even if a checkpoint
        # was written just before cleanup hung. Keep confirmed OOM/error evidence.
        if state.status not in ("oom", "error"):
            state.status = "error" if reason.startswith("supervisor error") else "incomplete"
        state.detail = reason + "; " + state.detail
    elif returncode != 0 and state.status not in ("oom", "error"):
        state.status = "error"
        state.detail = f"runner exited with code {returncode}; " + state.detail
    report = build_report(plan, state, predictions)
    _atomic_json(output / "report.json", report)
    (output / "summary.txt").write_text(summary(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args.request, args.adapter, args.output)
    report = supervise(args.request, args.adapter, args.output)
    print(summary(report), end="")
    return 0 if report["execution"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
