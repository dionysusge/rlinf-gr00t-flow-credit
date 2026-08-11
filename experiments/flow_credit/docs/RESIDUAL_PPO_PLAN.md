# GR00T N1.7 Residual PPO: E0 → E1 experiment plan

## Research question

Test whether a frozen GR00T N1.7 policy can be improved by an
action-conditioned Gaussian correction policy with less destructive drift than
full Flow-SDE PPO. The correction explicitly sees the complete normalized
GR00T proposal:

```text
u = f(mean_pool(h_VLM), h_state, vec(a_GR00T[1:16, 1:7]))
delta_a = 0.1 * tanh(u)
a_exec = a_GR00T + delta_a
```

There is no second activation on `a_exec`. The bound is a diagnostic trust
region in normalized correction space, not a representational requirement.
The first round answers only:

1. Can Residual PPO improve fixed-reset LIBERO-Spatial success?
2. Does it avoid the intermediate collapse observed in full PPO?
3. Does it learn small, sparse, temporally/action-dimension structured changes?

## Fixed setup

- Historical unseeded base reference: GR00T N1.7 LIBERO-Spatial SFT
  (444/500, 88.8%); the seeded E0 run establishes the new paired base.
- Full PPO reference: global step 600 (458/500, 91.6%).
- Hardware lanes: seeded evaluation jobs use physical H200 GPUs 4 and 5;
  Residual PPO training uses physical H200 GPUs 2 and 3.
- Action horizon: 16; flow steps: 4.
- Residual: normalized-action bound 0.1, two-layer 512-width MLP,
  `log_std=-2.5`, zero-initialized mean head. Its input is pooled VLM features,
  state features and the flattened normalized 16x7 GR00T action chunk.
- PPO: clip 0.2, gamma 0.99, GAE lambda 0.95, actor/critic LR 1e-4.
- Trainable modules: residual actor and value head only. GR00T, its VLM and
  flow action head remain frozen and in eval mode.

The old/new PPO ratio is computed only from the residual raw Gaussian sample,
conditioned on the exact GR00T proposal cached during rollout. PPO updates must
never recompute or omit that proposal when evaluating the recorded sample.
No Flow-SDE log-probability participates in the Residual PPO loss, and external
action noise is disabled.

## Server preparation

```bash
cd /data/Wayne/gzw/rlinf_gr00t_n17/RLinf
git fetch origin
git switch exp/policy-decorator-gr00t
git pull --ff-only
source /data/Wayne/gzw/rlinf_gr00t_n17/scripts/activate_rlinf.sh
```

Run long jobs inside `tmux`. E0 remains internally parallel on GPUs 4/5: Ray
launches two actor ranks, two rollout ranks and the parallel environment
workers required by the existing fixed-evaluation pipeline. Do not launch E1
on GPUs 2/3 until E0 has finished.
Every script validates its short `RAY_TMPDIR` before loading the model. The
fixed-evaluation entry shuts down its own local Ray runtime; the scripts do not
use the account-wide `ray stop --force` command. The experiment scripts also
set `RLINF_FORCE_LOCAL_RAY=1`. E0 additionally holds a launcher lock so a
second E0 cannot accidentally create another Ray runtime on GPUs 4/5.

## E0: seeded zero-residual equivalence check

```bash
tmux new -s residual-e0
bash experiments/flow_credit/scripts/run_n17_residual_e0_fixed500.sh
```

Flow inference samples an initial Gaussian latent even when language/token
sampling is disabled. `env.eval.seed` controls fixed LIBERO reset states;
`rollout.seed=1234` separately controls that model-side RNG (with a rank
offset). The five non-overlapping 100-trial slices start at offsets 0, 100, 200,
300 and 400. The script evaluates both residual-disabled and `force_zero=true`
policies on all five 100-trial sets, then creates `E0_PASS` only if all 500
paired success outcomes match exactly (`rescue=0`, `harm=0`).

It also runs E0-R: an independent second zero-residual Set-A evaluation.
`E0_REPEATABLE` means all 100 outcomes agree. This is now a hard gate because
per-trial rescue/harm claims are not meaningful if the seeded evaluator cannot
repeat them. The historical unseeded 444/500 result remains in
`historical_reference.json`; it is a reference, not an exact gate for the new
seeded inference stream. Do not launch E1 if E0 fails.

E0 runs in parallel on GPUs 4 and 5, but it runs as the only experiment. A
failed worker now has a 15-minute collective timeout instead of the former
three-hour Gloo wait; each fixed set has a 45-minute outer timeout and up to two
fresh-Ray attempts. Completed sets are validated and can be reused by setting
`E0_RESUME_ROOT` to the interrupted output root.

The E0 W&B run is registered at startup and retains a stable run ID. Every
completed set appends its fixed-slice success, reward and episode-length
metrics; final aggregate, pairing, repeatability, Tables and compact evidence
artifacts are uploaded to that same run in `GR00T-Residual-RL`.

## E1: Residual-PPO-0.1

```bash
tmux new -s residual-e1
bash experiments/flow_credit/scripts/run_n17_residual_ppo_train_gpu23.sh
```

The launcher validates the GR00T checkpoint and resolves the Cosmos backbone
from the two known server locations before Ray starts. Override them with
`PPO_MODEL_PATH` and `PPO_BACKBONE_MODEL_PATH` when moving the experiment.
LIBERO chunk rewards are masked after each environment's first done, so PPO
advantages never include post-terminal actions from the remainder of a 16-step
open-loop chunk.

One update contains 4096 environment transitions. The 150-step run saves and
evaluates checkpoints near 123K, 246K, 369K, 492K and 614K transitions (steps
30, 60, 90, 120 and 150). W&B/TensorBoard
logs include cumulative transitions, completed episodes, residual L2/quantiles,
active fractions, each action dimension, each of 16 chunk horizons, and the
sample and deterministic-mean residual/base norm ratios. They also separate
sampled correction, deterministic policy-mean correction and exploration
correction; record Gaussian `log_std`,
sample and policy-mean 0.09 saturation fractions, raw sample and raw mean
constraint pressure above `atanh(0.9)=1.472`, normalized-action OOD fractions, PPO ratio
mean/p95/max, clip fraction, approximate KL, actor/critic losses, gradient norm
and advantage standard deviation. All curves use
`progress/environment_transitions`, with `progress/global_step` aligned to
checkpoint/evaluation names.

Each run additionally writes `run_manifest.json` (host, GPU, versions, seeds,
paths, branch/commit and command) and a fully expanded `resolved_config.yaml`.
The training run uses a stable W&B run ID and appends both files as a compact
evidence artifact after training completes.

## Set-A checkpoint sweep

```bash
tmux new -s residual-sweep
bash experiments/flow_credit/scripts/run_n17_residual_checkpoint_sweep_setA.sh
```

For every checkpoint this produces:

- `metrics.json`, `trials.csv`, `trials.jsonl`;
- preserve/rescue/harm/unresolved pairing and exact McNemar test;
- default-on raw residual NPZ shards and step, dimension, horizon,
  mean/noise, sample/mean saturation, raw constraint pressure, OOD and
  success-conditioned analysis;
- `per_trial_residual.csv`, `high_pressure_trials.csv`, and
  `transition_conditioned.csv`, joined to preserve/rescue/harm/unresolved;
- `step_records.csv` as the ordered per-action-call trajectory, plus
  `per_episode_time.csv` and `per_task_episode_time.csv` using explicit
  early/middle/late temporal proxy bins;
- `per_task_residual.csv` for task heterogeneity and
  `per_decoded_dimension.csv` / `per_environment_dimension.csv` for the
  correction after GR00T unnormalization/relative-action decode and after the
  final LIBERO gripper command conversion, respectively;
- `checkpoint_summary.csv` indexed by both optimizer step and environment
  transitions;
- `checkpoint_selection_ranking.csv`, `selection.json` and
  `best_checkpoint.txt`.

Checkpoint selection is pre-registered: maximize Set-A success; on ties choose
fewer harms, then shorter episode length, then the earlier checkpoint. Set A is
development-only. Do not delete raw diagnostic NPZ files before analysis is
complete. Set `DUMP_RESIDUAL_DIAGNOSTICS=0` only for a deliberate low-storage
rerun.

`high_pressure_trials.csv` selects trials with deterministic mean saturation
above 10% or a maximum absolute mean correction above 0.095. These are the
cases to inspect before considering a 0.2 bound. Per-dimension and per-horizon
pressure distinguish a global radius limitation from rotation/gripper-specific
or short-horizon correction structure.

The temporal bins are computed from exact action-call order within each fixed
trial. They answer *when* the intervention occurs, but they are not semantic
phase labels such as approach/grasp/place. Use `step_records.csv` and the
task/trial/reset IDs in `high_pressure_trials.csv` to select trajectory or video
cases before making a semantic phase claim.

E0, checkpoint sweep, strength curve and Full-PPO paired evaluation each create
one run in the `GR00T-Residual-RL` W&B project under the
`Residual-Locality-Evidence` group. Success, rescue/harm, selection, saturation,
raw-pressure and temporal summary values are logged as metrics. Trial, task,
temporal, dimension, horizon, high-pressure and transition-conditioned CSVs are
logged as Tables and all compact structured files are retained as an artifact.
Raw NPZ shards are intentionally local-only.

## E2: residual strength curve

```bash
tmux new -s residual-strength
bash experiments/flow_credit/scripts/run_n17_residual_strength_curve_fixed500.sh \
  /absolute/path/to/checkpoints/global_step_N
```

This reuses one trained actor and evaluates lambda 0, 0.25, 0.5 and 1.0 over
the complete fixed 500 trials. It writes `strength_curve.csv` plus paired
rescue/harm/McNemar artifacts for every lambda. `heldout_BtoE/` and
`heldout_BtoE_pairing/` are the primary 400-trial held-out result; all-500
`aggregate/` and `pairing/` remain a descriptive aggregate because Set A was
used for checkpoint selection.
Lambda 0 is also an implementation gate: with the seeded model-side RNG it
must reproduce every E0 outcome exactly before the curve is accepted.

## Full PPO step600 paired baseline

After E0, re-evaluate the already-trained Full PPO step600 checkpoint on the
same fixed resets. This requires no retraining and distinguishes net gain from
destructive policy reshuffling:

```bash
tmux new -s fullppo-fixed500
bash experiments/flow_credit/scripts/run_n17_fullppo_step600_fixed500.sh \
  /absolute/path/to/full_ppo/checkpoints/global_step_600
```

The script produces all-500 and held-out B--E fixed evaluations, each paired to
E0 with preserve/rescue/harm/unresolved and exact McNemar statistics.

## Reporting protocol

- Development and checkpoint selection: Set A (100 trials).
- Primary held-out result: Sets B--E (400 trials).
- Descriptive aggregate: Sets A--E (500 trials).

Report success together with rescue, harm, net rescue and per-task pairing. Do
not interpret a small normalized residual as a small physical correction by
itself: use the automatically emitted `decoded_*` / `environment_*`
diagnostics and their per-dimension CSVs. The frozen base and executed action
are decoded separately through the same state-conditioned GR00T path; the
`environment_*` layer also applies LIBERO's final binary gripper conversion.

## Go / no-go

- Strong go: about 91.5% or higher, no full-PPO-style collapse, fewer harms,
  small residuals and visible temporal/dimension structure.
- Go: 90.5–91.5% with better stability, fewer harms, smaller deviation or
  faster improvement.
- No-go: about 89% without meaningful improvement, large full-episode takeover,
  or instability comparable to full PPO.

Do not add latent residuals, gates, SAC, multi-seed sweeps or other tasks until
the E0/E1 evidence is complete.
