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
zero-residual fixed500 check, then trains one bounded residual actor and records
paired rescue/harm outcomes plus per-dimension and per-horizon corrections.
