# Bounded smoke tests (S3 / issue #3)

This is the reusable integration layer for [pparitoshh/fitcheck#3](https://github.com/pparitoshh/fitcheck/issues/3).
It depends on the reviewed TrainingSpec commit
`65baaa9ed369584561bf04fae9359c905043b7c2` (PR #2, unmerged when this work began).
`feat/smoke-test` branches from `feat/training-spec`; the parent is unchanged.
No harness (M1), calculator (M3), or full setup generator (M4) exists yet.
No training stack or GPU measurement implementation is supplied here.

The modules use Python 3.10+ and the standard library. The orchestration CLI is
POSIX-only (Linux/macOS); generated jobs target a single Slurm node. All tests use
CPU mocks. Real GPU execution, FSDP coordination and cluster validation are pending.

## Generate alongside a full setup

This example only writes files; it does not submit or run anything. Replace the
account, partition and adapter with the full setup's actual values. The adapter
name below is an integration placeholder, **not an existing training runner**.

```python
from pathlib import Path
from fitcheck import TrainingSpec
from fitcheck.smoke import SmokeLimits, SmokePlan, write_smoke_job

original = TrainingSpec.from_json(Path("examples/training-spec.json").read_text())
# In M4, pass the existing directory containing the full training setup instead.
setup_dir = Path("runs/setup-demo")
setup_dir.mkdir(parents=True, exist_ok=True)
plan = SmokePlan(original, SmokeLimits(wall_seconds=300, warmup_steps=3, measured_steps=10))
script = write_smoke_job(
    setup_dir, plan,
    adapter="your_harness:make_runner",
    project_dir=str(Path.cwd().resolve()),
    account="YOUR_PROJECT", partition="YOUR_PARTITION", cpus_per_task=8,
    predictions=None,  # No calculator exists yet: leave predictions unavailable.
)
print(script)
```

Outputs are `smoke-test/smoke.sbatch` and `smoke-test/request.json` beneath the
setup directory. `smoke-test/` is created exclusively: any existing directory,
file or symlink raises `FileExistsError`. Full-job files are untouched. Shared
absolute paths are embedded; regenerate at the destination if moving the setup.

Check syntax with `bash -n runs/setup-demo/smoke-test/smoke.sbatch`. Generation and
syntax checking are safe offline. Job submission is a separate user action.
Before any real run, M4 must prepare its pinned uv environment and download the
specified model revision and dataset to shared storage on a login node. No modules
are loaded and no packages, Python interpreters, weights or datasets are downloaded
by this wrapper. The script requires `$WORK` and `$SCRATCH`, runs in the supplied
shared project directory, and creates an exclusive results directory in `$SCRATCH`.
It preserves existing HF cache settings from the full setup.

The script uses [uv offline/no-sync execution](https://docs.astral.sh/uv/reference/cli/)
and a [Slurm time limit and GPU allocation](https://slurm.schedmd.com/sbatch.html).
It requests one controller task and all `TrainingSpec.gpu_count` GPUs. The shared
runner must launch and coordinate the local ranks. CPUs, account and partition
are explicit caller choices; site-specific resource constraints still need M4.

## Limits and provenance

Defaults are a 300-second allocation, 3 warm-up **optimizer updates**, and 10
measured optimizer updates. The controller reserves up to 10 seconds for reporting
and termination. Its deadline includes adapter import, initialization, updates and
cleanup. It terminates the worker process group, escalating to SIGKILL if needed;
Slurm supplies the outer allocation cap, including uv startup. Queue wait is excluded.

`SmokePlan` changes only `max_steps` / `num_epochs`. Model/revision, method, sequence
length, micro-batch, accumulation, precision, optimizer, packing, quantization,
GPU count and all LoRA/FSDP settings remain identical. Total updates are capped at
the original duration; for short jobs warm-up is reduced to retain at least one
measurement. Such reductions are reported. Epoch-based jobs require a positive
`epoch_step_count` resolved externally for the **whole original duration** with
the same dataset/batching rules. This layer never profiles a dataset or guesses it.

The request and report retain both complete specs, their TrainingSpec fingerprints,
the exact before/after duration overrides, requested limits, effective step limits
and any externally resolved epoch count. Deserialization reconstructs and verifies
the plan rather than trusting supplied fingerprints. When a job is already short
enough, the specs/fingerprints can be identical.

## Runner interface to integrate with M1

The adapter is `module.path:factory_name`. Its callable receives
`(original_spec, smoke_spec)` and returns `fitcheck.smoke_runner.TrainingRunner`:

| Member | Contract |
|---|---|
| `spec` | The **actually applied** TrainingSpec; its fingerprint must match the smoke spec. |
| `identity()` | JSON object describing actual hardware, software and workload inputs (below). |
| `train_step()` | Execute forward + backward for every accumulation micro-step, then one optimizer update **on every rank**. Return `StepResult(seconds, forward_micro_steps, backward_micro_steps, optimizer_steps)` with per-rank counter tuples. A load-only runner fails validation. |
| `reset_measurements()` | Synchronize all ranks after warm-up; reset allocator peaks and restart the device-memory sampler. Called once before measured updates, including when warm-up is zero. |
| `peak_memory()` | Return one cumulative `GPUPeak` per GPU rank since reset. Do not sum GPUs or reset between updates. Unknown metrics are `None`, never invented zeros. |
| `close()` | Stop samplers and all local workers, freeing resources even after a failure. |

`StepResult.seconds` is synchronized wall-clock duration of a complete optimizer
update across all ranks, including accumulated forward/backward work and optimizer
work. Synchronize before/after timing; reduce rank times with max if timing per
rank. The controller takes the median **after warm-up**. Adapter counters attest
execution; only GPU integration tests can confirm the implementation does real work.

`GPUPeak(rank, allocated_bytes, reserved_bytes, device_sampled_bytes,
sample_interval_seconds, sample_count)` keeps three metrics separate:

- Allocated: `torch.cuda.max_memory_allocated`, bytes, post-warm-up peak.
- Reserved: `torch.cuda.max_memory_reserved`, bytes, post-warm-up peak (GOAL.md M1).
- Device: sampled `nvidia-smi memory.used` peak, converted from MiB to bytes by the
  adapter. Record the polling interval and actual count per GPU. This is whole-device
  usage, potentially including other processes, and sampling can miss brief peaks.

The original spec is also passed so that the runner can retain full-job scheduler,
optimizer hyperparameters, dataset/tokenization, attention kernels and checkpoint
policy; do not silently recompute these from the shortened duration. Record their
resolved values in an `execution_fingerprint`. Only the training stopping limit
changes. Preserve the full workload's offline environment and FSDP strategy. Do not
detach child processes or submit Slurm jobs from the adapter.

Translate confirmed device OOM exceptions to `fitcheck.smoke_report.TrainingOOM`.
Other exceptions become `error`; data exhaustion/deadline/interruption becomes
`incomplete`. Abrupt nonzero exits are errors, not inferred OOMs. Failures retain
metrics through the last completed update; failed-step peaks may be unavailable.

## Predictions and reports

`identity()` and prediction `identity` have the same structure:

- `hardware.gpus`: rank-ordered objects with `rank`, `name`, `total_memory_bytes`;
  include UUIDs/additional device properties when available. Multiple GPUs also
  require `hardware.interconnect`.
- `software`: `python`, `torch`, `cuda`, `driver`, `transformers`, `peft`,
  `bitsandbytes`, `flash_attn` versions. Explicit `not-used` is permitted when
  applicable; missing/unknown values are not matching evidence.
- `resolved_model_revision`, `dataset_fingerprint`, `execution_fingerprint`,
  `runner_revision`: actual immutable revision/content identities, not guessed
  from a mutable model alias. Revisions must be lowercase 40/64-character hex
  hashes; dataset/execution fingerprints must be lowercase SHA-256 hex digests.
  M1 still owns/agrees the content included in these fingerprints. Alias strings
  such as `main` are not sufficient evidence for comparison.

An optional prediction object contains `source`, a complete TrainingSpec in `spec`,
its `spec_fingerprint`, `identity`, and `metrics`. Metric keys are
`optimizer_step_seconds`, `gpu/<rank>/allocated_bytes`,
`gpu/<rank>/reserved_bytes`, and `gpu/<rank>/device_sampled_bytes`.
Each prediction has `estimate`, optional `lower`/`upper` bounds, a `definition`
object, and explicit `duration_invariant: true` when comparing a different duration.
Definitions must come from the estimator's real measurement contract, not be
relabeled to force a match. See `measured_metrics()` for the versioned definition
fields: unit, statistic, source, scope, measurement window and sampling metadata.

Comparisons require a completed full measurement window, the actually applied
spec, identical non-duration workload settings, matching fingerprints, complete
and identical environment identities, and identical metric definitions/units.
Matching is deliberately conservative (even different recorded GPU UUIDs prevent
a match). Duration changes require estimator opt-in per metric: a full-run peak
may not be duration-invariant. Available predictions/bands remain in the JSON even
when comparison is unavailable, with explicit reasons. Missing values stay null.
Only comparable upper-bound exceedances generate warnings. Full-training duration
is never compared to smoke elapsed time; no extrapolation/recalibration occurs.

Each exclusive run directory contains `report.json`, `summary.txt`, `runner.log`,
and an atomically updated `checkpoint.json`. Status is `completed`, `oom`, `error`
or `incomplete`. JSON includes raw step times, per-GPU peaks and sample counts,
units, identity, configuration provenance, predictions, comparability reasons and
warnings. The summary labels unavailable predictions and prints comparable bounds.
`A completed smoke test does not guarantee that the full job will succeed` always
appears. Long-run peaks, checkpoint I/O, convergence and future failures remain unknown.

A SIGKILL/node failure cannot guarantee finalization. The prewritten report remains
incomplete; the last atomic checkpoint retains progress. UV/interpreter failures
before the controller starts only appear in Slurm logs. Scratch results must be
copied to durable storage by the caller. No full job is automatically submitted.

## Validation and pending work

Run `python3 -S -m unittest discover -s tests -v`. Tests use synthetic CPU-only
adapters, including real supervised mock subprocesses for timeout/crash/interrupt
behavior; generated scripts are checked with `bash -n`.

Pending: maintainer agreement on this proposed M1/M4 interface; a shared training
runner adapter and its GPU instrumentation; calculator prediction metadata/bands;
full-setup integration and pinned uv environment; real single-GPU/FSDP runs,
device OOM and Slurm signal behavior; site validation (including `sbatch --test-only`
on Leonardo). No GPU training, model downloads or Slurm submission was performed.
