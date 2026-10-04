"""CPU-only smoke-job planning and Slurm generation; no training implementation."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import shlex

from .training_spec import TrainingSpec


def positive_int(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class SmokeLimits:
    wall_seconds: int = 300
    warmup_steps: int = 3
    measured_steps: int = 10

    def __post_init__(self):
        positive_int("wall_seconds", self.wall_seconds, 2)
        positive_int("warmup_steps", self.warmup_steps, 0)
        positive_int("measured_steps", self.measured_steps)
        if self.wall_seconds > 86400:
            raise ValueError("wall_seconds exceeds the one-day cluster limit")

    @property
    def execution_seconds(self) -> int:
        # Leave room for termination and final reporting before Slurm's hard cap.
        return self.wall_seconds - min(10, max(1, self.wall_seconds // 5))


@dataclass(frozen=True, slots=True)
class SmokePlan:
    original: TrainingSpec
    limits: SmokeLimits = SmokeLimits()
    epoch_step_count: int | None = None

    def __post_init__(self):
        if not isinstance(self.original, TrainingSpec) or not isinstance(self.limits, SmokeLimits):
            raise ValueError("a TrainingSpec and SmokeLimits are required")
        if self.original.num_epochs is not None:
            positive_int("epoch_step_count (resolved by the caller)", self.epoch_step_count)
        elif self.epoch_step_count is not None:
            raise ValueError("epoch_step_count is only valid for epoch-based training")

    @property
    def smoke(self) -> TrainingSpec:
        total = self.original.max_steps or self.epoch_step_count
        return replace(self.original, num_epochs=None,
                       max_steps=min(total, self.limits.warmup_steps + self.limits.measured_steps))

    @property
    def warmup_steps(self) -> int:
        return min(self.limits.warmup_steps, self.smoke.max_steps - 1)

    @property
    def measured_steps(self) -> int:
        return self.smoke.max_steps - self.warmup_steps

    def to_dict(self) -> dict:
        original, smoke = self.original.to_dict(), self.smoke.to_dict()
        return {
            "schema_version": 1, "original_spec": original, "smoke_spec": smoke,
            "original_fingerprint": self.original.fingerprint(),
            "smoke_fingerprint": self.smoke.fingerprint(),
            "overrides": {key: {"original": original[key], "smoke": value}
                          for key, value in smoke.items() if value != original[key]},
            "limits": asdict(self.limits), "epoch_step_count": self.epoch_step_count,
            "warmup_steps": self.warmup_steps, "measured_steps": self.measured_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SmokePlan":
        try:
            plan = cls(TrainingSpec.from_dict(data["original_spec"]),
                       SmokeLimits(**data["limits"]), data["epoch_step_count"])
            if json_text(data) != json_text(plan.to_dict()):
                raise ValueError("smoke plan fields/fingerprints do not match the resolved plan")
            return plan
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid smoke plan: {exc}") from exc


def validate_adapter(adapter: str) -> None:
    if not isinstance(adapter, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", adapter):
        raise ValueError("adapter must be module.path:factory_name")


def render_slurm(plan: SmokePlan, *, adapter: str, project_dir: str,
                 request_path: str, account: str, partition: str,
                 cpus_per_task: int) -> str:
    """One controller owns all ranks; the shared runner launches FSDP workers."""
    validate_adapter(adapter)
    positive_int("cpus_per_task", cpus_per_task)
    for name, value in (("account", account), ("partition", partition)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"invalid Slurm {name}")
    for name, value in (("project_dir", project_dir), ("request_path", request_path)):
        if not isinstance(value, str) or not Path(value).is_absolute() or "\n" in value or "\r" in value:
            raise ValueError(f"{name} must be an absolute path without newlines")
    hours, remainder = divmod(plan.limits.wall_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'''#!/bin/bash
#SBATCH --job-name=fitcheck-smoke
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{plan.smoke.gpu_count}
#SBATCH --time={hours:02d}:{minutes:02d}:{seconds:02d}

set -euo pipefail
umask 077
: "${{WORK:?Set WORK to shared project/model storage}}"
: "${{SCRATCH:?Set SCRATCH to writable run storage}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
cd -- {shlex.quote(project_dir)}
run_root=$(mktemp -d "${{SCRATCH%/}}/fitcheck-smoke.XXXXXXXX")
printf 'Smoke-test output: %s\\n' "$run_root/report"
# Install/pin the environment and pre-download weights/data on a login node first.
# No environment synchronization, model downloads, or full-job submission here.
uv run --offline --no-sync python -m fitcheck.smoke_runner \\
  --request {shlex.quote(request_path)} --adapter {shlex.quote(adapter)} \\
  --output "$run_root/report"
'''


def write_smoke_job(setup_dir: Path, plan: SmokePlan, *, adapter: str,
                    project_dir: str, account: str, partition: str,
                    cpus_per_task: int, predictions: dict | None = None) -> Path:
    """Add an exclusive smoke-test/ bundle alongside an existing full setup.

    Nothing is overwritten; a prior or partially generated bundle must be kept
    or explicitly removed by its owner before retrying. Does not submit a job.
    """
    setup_dir = Path(setup_dir).resolve(strict=True)
    if not setup_dir.is_dir():
        raise ValueError("setup_dir must be an existing directory")
    bundle = setup_dir / "smoke-test"
    request = json_text({"plan": plan.to_dict(), "predictions": predictions})
    script = render_slurm(plan, adapter=adapter, project_dir=project_dir,
                          request_path=str(bundle / "request.json"), account=account,
                          partition=partition, cpus_per_task=cpus_per_task)
    bundle.mkdir(mode=0o700)  # Exclusive even if the path is a symlink.
    (bundle / "request.json").write_text(request, encoding="utf-8")
    (bundle / "smoke.sbatch").write_text(script, encoding="utf-8")
    return bundle / "smoke.sbatch"
