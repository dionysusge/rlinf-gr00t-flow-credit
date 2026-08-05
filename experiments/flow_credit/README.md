# GR00T Action-Flow Credit Assignment

This directory contains experiments for studying non-uniform RL credit
assignment across the internal flow-denoising steps of GR00T N1.7.

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
