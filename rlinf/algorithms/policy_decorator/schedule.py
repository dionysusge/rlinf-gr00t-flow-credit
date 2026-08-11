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

"""Progressive residual exploration schedule."""

from __future__ import annotations

import torch


class ProgressiveResidualSchedule:
    """Linearly increase the fraction of environments using residual actions."""

    def __init__(self, warmup_steps: int) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_steps = int(warmup_steps)

    def ratio(self, step: int) -> float:
        """Return the residual activation probability for a decision step."""
        if self.warmup_steps == 0:
            return 1.0
        return min(max(float(step) / float(self.warmup_steps), 0.0), 1.0)

    def mask(
        self,
        *,
        step: int,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a ``[batch,1,1]`` boolean activation mask."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        probability = self.ratio(step)
        if probability <= 0.0:
            return torch.zeros((batch_size, 1, 1), dtype=torch.bool, device=device)
        if probability >= 1.0:
            return torch.ones((batch_size, 1, 1), dtype=torch.bool, device=device)
        return (
            torch.rand((batch_size, 1, 1), device=device, generator=generator)
            < probability
        )
