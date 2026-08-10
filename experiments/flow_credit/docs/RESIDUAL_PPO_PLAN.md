# GR00T N1.7 Residual PPO: E0 → E1 experiment plan

## Research question

Test whether a frozen GR00T N1.7 policy can be improved by a small Gaussian
residual actor with less destructive drift than full Flow-SDE PPO. The first
round answers only:

1. Can Residual PPO improve fixed-reset LIBERO-Spatial success?
2. Does it avoid the intermediate collapse observed in full PPO?
3. Does it learn small, sparse, temporally/action-dimension structured changes?

## Fixed setup

- Base: GR00T N1.7 LIBERO-Spatial SFT (444/500, 88.8%).
- Full PPO reference: global step 600 (458/500, 91.6%).
- Hardware: physical H200 GPUs 2 and 3.
- Action horizon: 16; flow steps: 4.
- Residual: normalized-action bound 0.1, two-layer 512-width MLP,
  `log_std=-2.5`, zero-initialized mean head.
- PPO: clip 0.2, gamma 0.99, GAE lambda 0.95, actor/critic LR 1e-4.
- Trainable modules: residual actor and value head only. GR00T, its VLM and
  flow action head remain frozen and in eval mode.

The old/new PPO ratio is computed only from the residual raw Gaussian sample.
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

Run long jobs inside `tmux`. Check GPU 2/3 occupancy before launching.

## E0: exact zero-residual check

```bash
tmux new -s residual-e0
bash experiments/flow_credit/scripts/run_n17_residual_e0_fixed500.sh
```

The script evaluates all five 100-trial sets with `force_zero=true`, writes
structured `trials.csv`/`trials.jsonl`, and creates `E0_PASS` only when all 500
trials complete and success is exactly 444/500. Do not launch E1 if E0 fails.

## E1: Residual-PPO-0.1

```bash
tmux new -s residual-e1
bash experiments/flow_credit/scripts/run_n17_residual_ppo_train_gpu23.sh
```

One update contains 4096 environment transitions. Checkpoints are saved near
50K, 100K, 150K and 200K transitions (steps 12, 24, 36 and 48). W&B/TensorBoard
logs include cumulative transitions, completed episodes, residual L2/quantiles,
active fractions, each action dimension, each of 16 chunk horizons, and the
residual/base norm ratio.

## Set-A checkpoint sweep

```bash
tmux new -s residual-sweep
DUMP_RESIDUAL_DIAGNOSTICS=1 \
  bash experiments/flow_credit/scripts/run_n17_residual_checkpoint_sweep_setA.sh
```

For every checkpoint this produces:

- `metrics.json`, `trials.csv`, `trials.jsonl`;
- preserve/rescue/harm/unresolved pairing and exact McNemar test;
- optional step, dimension, horizon and success-conditioned residual analysis;
- `checkpoint_summary.csv` indexed by both optimizer step and environment
  transitions.

Choose the best checkpoint from Set A before running fixed500.

## E2: residual strength curve

```bash
tmux new -s residual-strength
DUMP_RESIDUAL_DIAGNOSTICS=1 \
  bash experiments/flow_credit/scripts/run_n17_residual_strength_curve_fixed500.sh \
  /absolute/path/to/checkpoints/global_step_N
```

This reuses one trained actor and evaluates lambda 0, 0.25, 0.5 and 1.0 over
the complete fixed 500 trials. It writes `strength_curve.csv` plus paired
rescue/harm/McNemar artifacts for every lambda.

## Go / no-go

- Strong go: about 91.5% or higher, no full-PPO-style collapse, fewer harms,
  small residuals and visible temporal/dimension structure.
- Go: 90.5–91.5% with better stability, fewer harms, smaller deviation or
  faster improvement.
- No-go: about 89% without meaningful improvement, large full-episode takeover,
  or instability comparable to full PPO.

Do not add latent residuals, gates, SAC, multi-seed sweeps or other tasks until
the E0/E1 evidence is complete.
