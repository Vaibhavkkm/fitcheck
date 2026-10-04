"""Measurement contract and conservative predicted-vs-measured comparisons."""

from dataclasses import asdict, dataclass, field
import math
import re
import statistics

from .smoke import SmokePlan, json_text, positive_int
from .training_spec import TrainingSpec


class TrainingOOM(RuntimeError):
    """Adapters translate a confirmed device OOM to this shared exception type."""


def nonnegative(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True, slots=True)
class StepResult:
    """One synchronized optimizer update, including every accumulation micro-step."""

    seconds: float
    forward_micro_steps: tuple[int, ...]
    backward_micro_steps: tuple[int, ...]
    optimizer_steps: tuple[int, ...]

    def validate(self, spec: TrainingSpec):
        nonnegative("step seconds", self.seconds)
        if self.seconds == 0:
            raise ValueError("training-step time must be positive")
        for counts, expected in ((self.forward_micro_steps, spec.gradient_accumulation_steps),
                                 (self.backward_micro_steps, spec.gradient_accumulation_steps),
                                 (self.optimizer_steps, 1)):
            if len(counts) != spec.gpu_count or any(type(v) is not int or v != expected for v in counts):
                raise ValueError("runner must execute forward/backward/optimizer steps on every rank")


@dataclass(frozen=True, slots=True)
class GPUPeak:
    rank: int
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    device_sampled_bytes: int | None = None
    sample_interval_seconds: float | None = None
    sample_count: int = 0

    def __post_init__(self):
        positive_int("GPU rank", self.rank, 0)
        positive_int("sample_count", self.sample_count, 0)
        for name in ("allocated_bytes", "reserved_bytes", "device_sampled_bytes"):
            if getattr(self, name) is not None:
                positive_int(name, getattr(self, name), 0)
        if self.allocated_bytes is not None and self.reserved_bytes is not None:
            if self.allocated_bytes > self.reserved_bytes:
                raise ValueError("allocated peak cannot exceed reserved peak in the same window")
        if self.device_sampled_bytes is None:
            if self.sample_interval_seconds is not None or self.sample_count:
                raise ValueError("sampling metadata requires a sampled device peak")
        elif (type(self.sample_interval_seconds) not in (int, float)
              or not math.isfinite(self.sample_interval_seconds)
              or self.sample_interval_seconds <= 0 or not self.sample_count):
            raise ValueError("sampled device memory needs a positive interval and sample count")


@dataclass
class RunState:
    status: str = "incomplete"
    detail: str = "Runner has not completed."
    identity: dict | None = None
    applied_spec_fingerprint: str | None = None
    warmup_completed: int = 0
    step_seconds: list[float] = field(default_factory=list)
    gpu_peaks: list[GPUPeak] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        data["gpu_peaks"] = [GPUPeak(**item) for item in data["gpu_peaks"]]
        return cls(**data)


def measured_metrics(state: RunState, gpu_count: int) -> dict:
    def metric(value, unit, statistic, source, scope, sampled=False, interval=None):
        return {"value": value, "definition": {
            "unit": unit, "statistic": statistic, "source": source, "scope": scope,
            "window": "post_warmup_optimizer_updates", "sampled": sampled,
            "sample_interval_seconds": interval,
        }}
    result = {"optimizer_step_seconds": metric(
        statistics.median(state.step_seconds) if state.step_seconds else None,
        "s", "median", "synchronized_wall_clock", "all_ranks")}
    peaks = {item.rank: item for item in state.gpu_peaks}
    for rank in range(gpu_count):
        gpu = peaks.get(rank, GPUPeak(rank))
        for field_name, source, sampled in (
            ("allocated_bytes", "torch.cuda.max_memory_allocated", False),
            ("reserved_bytes", "torch.cuda.max_memory_reserved", False),
            ("device_sampled_bytes", "nvidia-smi.memory.used", True),
        ):
            result[f"gpu/{rank}/{field_name}"] = metric(
                getattr(gpu, field_name), "bytes", "peak", source, f"gpu/{rank}",
                sampled, gpu.sample_interval_seconds if sampled else None)
    return result


def identity_reasons(identity: dict | None, gpu_count: int) -> list[str]:
    """Missing values never become a match merely because both sides omit them."""
    if not isinstance(identity, dict):
        return ["hardware/software/workload identity unavailable"]
    reasons = []
    def known(value):
        return isinstance(value, str) and value.strip().lower() not in ("", "unknown", "unavailable", "none", "null", "n/a")
    for key in ("resolved_model_revision", "dataset_fingerprint", "execution_fingerprint", "runner_revision"):
        value = identity.get(key)
        pattern = r"[0-9a-f]{64}" if key.endswith("fingerprint") else r"(?:[0-9a-f]{40}|[0-9a-f]{64})"
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            reasons.append(f"identity.{key} requires an immutable content hash")
    software = identity.get("software", {})
    for key in ("python", "torch", "cuda", "driver", "transformers", "peft", "bitsandbytes", "flash_attn"):
        if not isinstance(software, dict) or not known(software.get(key)):
            reasons.append(f"software.{key} unavailable (use not-used only when applicable)")
    hardware = identity.get("hardware", {})
    gpus = hardware.get("gpus") if isinstance(hardware, dict) else None
    if not isinstance(gpus, list) or len(gpus) != gpu_count:
        reasons.append("GPU hardware identity/count unavailable")
    else:
        for rank, gpu in enumerate(gpus):
            if (not isinstance(gpu, dict) or type(gpu.get("rank")) is not int or gpu["rank"] != rank
                    or not known(gpu.get("name"))
                    or type(gpu.get("total_memory_bytes")) is not int
                    or gpu["total_memory_bytes"] <= 0):
                reasons.append(f"GPU {rank} hardware identity unavailable")
    if gpu_count > 1 and (not isinstance(hardware, dict) or not known(hardware.get("interconnect"))):
        reasons.append("multi-GPU interconnect identity unavailable")
    return reasons


def build_report(plan: SmokePlan, state: RunState, predictions: dict | None = None) -> dict:
    """Compare like metrics only; full-run total duration is never compared."""
    measured = measured_metrics(state, plan.smoke.gpu_count)
    shared_reasons = []
    predicted_metrics = {}
    duration_changed = False
    if state.status != "completed":
        shared_reasons.append(f"execution is {state.status}; partial measurements are not comparable")
    if state.warmup_completed != plan.warmup_steps or len(state.step_seconds) != plan.measured_steps:
        shared_reasons.append("requested warm-up/measurement window is incomplete")
    if state.applied_spec_fingerprint != plan.smoke.fingerprint():
        shared_reasons.append("runner's applied TrainingSpec fingerprint is unavailable or mismatched")
    shared_reasons.extend(identity_reasons(state.identity, plan.smoke.gpu_count))
    if predictions is None:
        shared_reasons.append("predictions unavailable")
    else:
        try:
            if not isinstance(predictions.get("source"), str) or not predictions["source"].strip():
                shared_reasons.append("prediction source unavailable")
            predicted_spec = TrainingSpec.from_dict(predictions["spec"])
            if predicted_spec.fingerprint() != predictions.get("spec_fingerprint"):
                shared_reasons.append("prediction specification fingerprint mismatch")
            actual, predicted = plan.smoke.to_dict(), predicted_spec.to_dict()
            duration_changed = any(actual[k] != predicted[k] for k in ("max_steps", "num_epochs"))
            for key in ("max_steps", "num_epochs"):
                actual.pop(key)
                predicted.pop(key)
            if actual != predicted:
                shared_reasons.append("prediction workload differs beyond the intentional duration override")
            shared_reasons.extend("prediction " + reason for reason in
                                  identity_reasons(predictions.get("identity"), plan.smoke.gpu_count))
            if state.identity != predictions.get("identity"):
                shared_reasons.append("prediction hardware/software/workload identity mismatch")
            predicted_metrics = predictions["metrics"]
            if not isinstance(predicted_metrics, dict):
                raise ValueError("metrics must be an object")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            shared_reasons.append(f"invalid predictions: {exc}")
            predicted_metrics = {}
    warnings = ["A completed smoke test does not guarantee that the full job will succeed."]
    if plan.warmup_steps != plan.limits.warmup_steps:
        warnings.append("Warm-up was shortened to avoid extending the original training duration.")
    comparisons = {}
    for name in sorted(measured.keys() | predicted_metrics.keys()):
        actual = measured.get(name)
        prediction = predicted_metrics.get(name)
        reasons = list(shared_reasons)
        if actual is None:
            reasons.append("unsupported metric; full-training duration is never compared to smoke elapsed time")
        elif actual["value"] is None:
            reasons.append("measurement unavailable")
        if prediction is None:
            reasons.append("prediction unavailable for this metric")
        else:
            try:
                if actual is not None and prediction["definition"] != actual["definition"]:
                    reasons.append("metric definition/units/sampling mismatch")
                nonnegative("prediction estimate", prediction["estimate"])
                for bound in ("lower", "upper"):
                    if prediction.get(bound) is not None:
                        nonnegative(f"prediction {bound}", prediction[bound])
                if (prediction.get("lower") is not None and prediction["lower"] > prediction["estimate"]
                        or prediction.get("upper") is not None and prediction["upper"] < prediction["estimate"]):
                    raise ValueError("prediction estimate lies outside its bounds")
                if duration_changed and prediction.get("duration_invariant") is not True:
                    reasons.append("prediction does not declare this metric invariant to duration changes")
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                reasons.append(f"invalid metric prediction: {exc}")
        comparable = not reasons
        exceeds = (actual["value"] > prediction["upper"]
                   if comparable and prediction.get("upper") is not None else None)
        if exceeds:
            warnings.append(f"{name}: comparable measurement exceeds predicted upper bound.")
        comparisons[name] = {"status": "comparable" if comparable else "unavailable",
                             "reasons": reasons, "measurement": actual, "prediction": prediction,
                             "exceeds_upper_bound": exceeds}
    report = {"schema_version": 1, "plan": plan.to_dict(), "execution": state.to_dict(),
              "metrics": measured, "predictions": predictions, "comparisons": comparisons,
              "warnings": warnings, "elapsed_time_unit": "s"}
    json_text(report)  # Fail on non-JSON identity/NaN instead of writing corrupt reports.
    return report


def summary(report: dict) -> str:
    execution = report["execution"]
    lines = [f"Smoke test: {execution['status']} — {execution['detail']}",
             f"Original fingerprint: {report['plan']['original_fingerprint']}",
             f"Smoke fingerprint: {report['plan']['smoke_fingerprint']}",
             f"Hardware/software identity: {json_text(execution['identity']).strip()}",
             f"Warm-up updates: {execution['warmup_completed']}; measured updates: {len(execution['step_seconds'])}",
             f"Elapsed: {execution['elapsed_seconds']:.3f} s (not compared to full-training duration)"]
    for name, comparison in report["comparisons"].items():
        metric, prediction = comparison["measurement"], comparison["prediction"]
        actual = "unavailable" if metric is None or metric["value"] is None else f"{metric['value']} {metric['definition']['unit']}"
        def available(key):
            return prediction.get(key) if prediction.get(key) is not None else "unavailable"
        predicted = (f"{available('estimate')} "
                     f"[lower={available('lower')}, upper={available('upper')}]"
                     if isinstance(prediction, dict) else "unavailable")
        lines.append(f"{name}: measured={actual}; predicted={predicted}; {comparison['status']}")
        if comparison["reasons"]:
            lines.append("  " + "; ".join(comparison["reasons"]))
    lines.extend("WARNING: " + item for item in report["warnings"])
    return "\n".join(lines) + "\n"
