# GR00T online-RL diagnostics and residual refinement

This directory preserves the original flow-credit diagnostics and hosts the
current frozen-policy Residual PPO experiments for GR00T N1.7.

Initial experiment groups:

1. Uniform timestep credit
2. Early-heavy credit
3. Middle-heavy credit
4. Late-heavy credit
5. Flow-step causal intervention
6. Dynamic value-gain credit
7. Vector-field preservation
8. Timestep-gated adapters

The initial benchmark is LIBERO-Spatial using the existing RLinf Flow-SDE PPO
training and fixed-reset evaluation pipeline.

The current experiment is documented in
[`docs/RESIDUAL_PPO_PLAN.md`](docs/RESIDUAL_PPO_PLAN.md). It starts with an exact
zero-residual fixed500 check, then trains one bounded, action-conditioned
correction actor. The actor receives pooled VLM features, state features and the
full normalized GR00T action proposal; it executes
`a_GR00T + 0.1 * tanh(raw_correction)` without activating the sum again. The
pipeline records paired rescue/harm outcomes, per-dimension and per-horizon
corrections, sample/mean saturation, raw constraint pressure and trial-level
high-pressure cases. The diagnostic decode path additionally records continuous
decoded actions and the final LIBERO action representation (including binary
gripper conversion), plus per-trial timing, early/middle/late temporal proxies
and per-task views. It
also records Set-A repeatability, a pre-registered Set-A selection rule, B--E
held-out results and a paired Full-PPO step600 reference.
E0 preserves the existing Ray/FSDP evaluation architecture and runs internally
in parallel on physical GPUs 4 and 5. Its five 100-trial slices use offsets
0/100/200/300/400, producing 500 unique task/trial pairs. A launcher lock,
short collective timeout, bounded retry and resumable per-set validation keep a
single worker failure from turning into a three-hour Gloo hang.
Training curves and every evaluation stage are synchronized to the
`GR00T-Residual-RL` W&B project. Compact CSV/JSON/YAML evidence is uploaded as
an artifact and the key trial files are exposed as W&B Tables; raw diagnostic
NPZ shards, videos and logs remain on the server.
