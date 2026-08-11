# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for converting open-loop action chunks into valid transitions."""

from __future__ import annotations

import torch


def mask_after_first_done(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask steps after each environment's first terminal chunk element.

    Vector environments cannot stop the whole action chunk when only one
    environment finishes. They may still advance that simulator until the
    batch reaches the chunk boundary, but those post-terminal samples must not
    become part of the RL transition.

    Args:
        rewards: Per-step rewards with shape ``[num_envs, chunk_steps]``.
        terminations: Per-step terminal flags with the same shape.
        truncations: Per-step time-limit flags with the same shape.

    Returns:
        Masked rewards, first-only terminations, first-only truncations, and a
        boolean mask identifying valid steps (including the first done step).
    """
    if not (
        rewards.ndim == terminations.ndim == truncations.ndim == 2
        and rewards.shape == terminations.shape == truncations.shape
    ):
        raise ValueError(
            "rewards, terminations, and truncations must share [env, step] shape"
        )
    terminations = terminations.bool()
    truncations = truncations.bool()
    dones = terminations | truncations
    prior_done = torch.zeros_like(dones)
    if dones.shape[1] > 1:
        prior_done[:, 1:] = dones[:, :-1].cumsum(dim=1) > 0
    valid_steps = ~prior_done
    return (
        rewards * valid_steps.to(dtype=rewards.dtype),
        terminations & valid_steps,
        truncations & valid_steps,
        valid_steps,
    )
