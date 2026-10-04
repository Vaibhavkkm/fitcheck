"""CPU mocks only. These values are synthetic, never GPU measurements."""

import os
import time

from fitcheck.smoke_report import GPUPeak, StepResult
from fitcheck.smoke_runner import TrainingOOM


class FakeRunner:
    def __init__(self, original, smoke, *, failure=None):
        self.original, self.spec, self.failure = original, smoke, failure
        self.steps, self.reset_at, self.closed = 0, None, False
        self.events = []

    def identity(self):
        return {
            "hardware": {"gpus": [{"rank": rank, "name": "MOCK-GPU",
                                    "total_memory_bytes": 64000}
                                   for rank in range(self.spec.gpu_count)],
                         "interconnect": "mock"},
            "software": {key: "mock" for key in
                         ("python", "torch", "cuda", "driver", "transformers", "peft", "bitsandbytes", "flash_attn")},
            "resolved_model_revision": "a" * 40, "dataset_fingerprint": "b" * 64,
            "execution_fingerprint": "c" * 64, "runner_revision": "d" * 40,
        }

    def train_step(self):
        self.events.append("step")
        if self.failure == "hang":
            time.sleep(60)
        if self.failure == "crash":
            os._exit(7)
        if self.steps == 2:
            if self.failure == "oom":
                raise TrainingOOM("synthetic GPU OOM")
            if self.failure == "error":
                raise RuntimeError("synthetic runner failure")
            if self.failure == "short":
                raise StopIteration
        self.steps += 1
        count = self.spec.gpu_count
        micro = self.spec.gradient_accumulation_steps
        return StepResult(float(self.steps - self.reset_at) if self.reset_at is not None else 999.0,
                          (micro,) * count, (micro,) * count,
                          (0 if self.failure == "load_only" else 1,) * count)

    def reset_measurements(self):
        self.events.append("reset")
        self.reset_at = self.steps

    def peak_memory(self):
        self.events.append("peaks")
        measured = self.steps - self.reset_at
        return [GPUPeak(rank, 100 + rank + measured, 200 + rank + measured,
                        300 + rank + measured, 0.1, measured)
                for rank in range(self.spec.gpu_count)]

    def close(self):
        self.events.append("close")
        self.closed = True
        if self.failure == "close_hang":
            time.sleep(60)


def ok(original, smoke):
    return FakeRunner(original, smoke)


def oom(original, smoke):
    return FakeRunner(original, smoke, failure="oom")


def hang(original, smoke):
    return FakeRunner(original, smoke, failure="hang")


def init_hang(original, smoke):
    time.sleep(60)
    return FakeRunner(original, smoke)


def close_hang(original, smoke):
    return FakeRunner(original, smoke, failure="close_hang")


def crash(original, smoke):
    return FakeRunner(original, smoke, failure="crash")
