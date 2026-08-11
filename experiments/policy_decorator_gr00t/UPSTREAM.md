# Method provenance

This implementation is an independent RLinf integration inspired by:

- Paper: *Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model*, ICLR 2025, <https://arxiv.org/abs/2412.13630>
- Official repository: <https://github.com/tongzhoumu/policy_decorator>
- Repository revision reviewed while designing the integration: `92ba9ba442587ae5989c286355ace2200a0537fb`
- Principal files reviewed for algorithm behavior:
  - <https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2.py>
  - <https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2_rgbd.py>

No upstream source file was copied verbatim. The official repository README states MIT, but its `LICENSE` file was empty at the reviewed revision, and the repository includes third-party components. The RLinf code here was therefore written independently against the paper/repository behavior and is distributed under RLinf's Apache-2.0 license.

The integration preserves the method-level ideas needed for this experiment:

- action-conditioned residual actor;
- bounded additive correction around a frozen base proposal;
- twin-Q SAC with entropy autotuning and target critics;
- replay-buffer training;
- progressive residual exploration;
- deterministic mean residual at evaluation time.

RLinf's current upstream RLT implementation was also reviewed as an architectural reference, not treated as the same method. RLT depends on a stage-1 RL token and edits a reference chunk, whereas this experiment uses a strict additive residual budget around the original GR00T proposal.
