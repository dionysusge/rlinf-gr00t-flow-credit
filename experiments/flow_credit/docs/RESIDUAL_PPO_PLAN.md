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

- Base: GR00T N1.7 LIBERO-Spatial SFT (444/500, 88.8%).
- Full PPO reference: global step 600 (458/500, 91.6%).
- Hardware: physical H200 GPUs 0 and 1.
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
git switch exp/residual-ppo-e0-e1
git pull --ff-only
source /data/Wayne/gzw/rlinf_gr00t_n17/scripts/activate_rlinf.sh
```

Run long jobs inside `tmux`. Check GPU 0/1 occupancy before launching.

## E0: exact zero-residual check

```bash
tmux new -s residual-e0
bash experiments/flow_credit/scripts/run_n17_residual_e0_fixed500.sh
```

The script evaluates all five 100-trial sets with `force_zero=true`, writes
structured `trials.csv`/`trials.jsonl`, and creates `E0_PASS` only when all 500
trials complete and success is exactly 444/500. Do not launch E1 if E0 fails.

It also runs E0-R: an independent second Set-A evaluation and paired analysis
against the first Set-A run. Inspect `E0_REPEATABILITY.json` and
`setA_repeatability/`: `E0_REPEATABLE` means every one of the 100 success
outcomes agreed. A mismatch creates `E0_REPEATABILITY_WARNING` but does not
replace the primary 444/500 gate; it makes any later rescue/harm uncertainty
explicit.

## E1: Residual-PPO-0.1

```bash
tmux new -s residual-e1
bash experiments/flow_credit/scripts/run_n17_residual_ppo_train_gpu01.sh
```

One update contains 4096 environment transitions. The 150-step run saves and
evaluates checkpoints near 123K, 246K, 369K, 492K and 614K transitions (steps
30, 60, 90, 120 and 150). W&B/TensorBoard
logs include cumulative transitions, completed episodes, residual L2/quantiles,
active fractions, each action dimension, each of 16 chunk horizons, and the
residual/base norm ratio. They also separate sampled correction, deterministic
policy-mean correction and exploration correction; record Gaussian `log_std`,
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

E0, checkpoint sweep, strength curve and Full-PPO paired evaluation each create
one run in the `GR00T-Residual-RL` W&B project under the
`Residual-Locality-Evidence` group. Success, rescue/harm, selection, saturation
and raw-pressure values are logged as metrics. Trial, high-pressure and
transition-conditioned CSVs are logged as Tables and all compact structured
files are retained as an artifact. Raw NPZ shards are intentionally local-only.

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
not interpret a small normalized residual as a small physical correction until
the best checkpoint has been re-dumped with decoded physical-space action
statistics.

## Go / no-go

- Strong go: about 91.5% or higher, no full-PPO-style collapse, fewer harms,
  small residuals and visible temporal/dimension structure.
- Go: 90.5–91.5% with better stability, fewer harms, smaller deviation or
  faster improvement.
- No-go: about 89% without meaningful improvement, large full-episode takeover,
  or instability comparable to full PPO.

Do not add latent residuals, gates, SAC, multi-seed sweeps or other tasks until
the E0/E1 evidence is complete.
