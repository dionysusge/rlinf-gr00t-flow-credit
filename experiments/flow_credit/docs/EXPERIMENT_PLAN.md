# Flow Credit Experiment Plan

## Research question

How should RL credit be assigned across the internal flow-denoising timesteps
of a GR00T N1.7 action policy?

## Baseline

- RLinf v0.3 server snapshot
- GR00T N1.7 LIBERO-Spatial SFT checkpoint
- Flow-SDE PPO
- Action horizon: 16
- Denoising steps: 4
- LIBERO-Spatial
- Fixed-reset evaluation protocol

## Phase 1: Diagnostics

Record per-flow-step:

- flow timestep
- latent transition norm
- log probability
- PPO ratio
- approximate KL
- clip fraction
- action change
- gradient norm or gradient proxy

## Phase 2: Static weighting

Compare:

- uniform
- early-heavy
- middle-heavy
- late-heavy
- U-shaped

All weight vectors must be normalized to mean 1.

## Phase 3: Causal intervention

Branch the denoising trajectory at each flow step and measure:

- final action difference
- low-frequency trajectory difference
- high-frequency trajectory difference
- end-effector pose difference
- gripper timing difference
- task success variance

## Phase 4: Adaptive credit

Candidate methods:

- transition-magnitude weighting
- value-gain weighting
- causal-influence weighting
- vector-field preservation
- timestep-gated adapters

## Evaluation

Use identical:

- initial SFT checkpoint
- environment reset states
- task/trial IDs
- random seeds
- number of PPO updates
- optimizer settings
- fixed evaluation protocol
