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

"""Thin tensor boundary between frozen GR00T and standalone residual SAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _regular_float_tensor(
    value: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Copy a possibly inference-mode tensor into ordinary learner storage."""
    array = value.detach().float().cpu().numpy().copy()
    tensor = torch.from_numpy(array).to(device=device)
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Non-finite tensor crossing the GR00T adapter")
    return tensor


@dataclass(frozen=True)
class GR00TDecoratorProposal:
    """Learner inputs plus opaque model-side data needed for decoding."""

    context: torch.Tensor
    base_action: torch.Tensor
    model_payload: dict[str, Any]


class GR00TPolicyDecoratorAdapter:
    """Freeze GR00T and expose proposal/decode operations only."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        learner_device: torch.device | str,
    ) -> None:
        if bool(getattr(model, "residual_policy_enabled", False)):
            raise RuntimeError(
                "The standalone adapter refuses a GR00T model with an embedded "
                "residual policy"
            )
        self.model = model
        self.learner_device = torch.device(learner_device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def propose(self, env_obs: dict[str, Any]) -> GR00TDecoratorProposal:
        """Return an ordinary float32 context and normalized base chunk."""
        payload = self.model.get_policy_decorator_proposal(env_obs)
        context = _regular_float_tensor(payload["context"], device=self.learner_device)
        base_action = _regular_float_tensor(
            payload["base_action"], device=self.learner_device
        )
        return GR00TDecoratorProposal(
            context=context,
            base_action=base_action,
            model_payload=payload,
        )

    @torch.no_grad()
    def decode(
        self,
        proposal: GR00TDecoratorProposal,
        executed_normalized_action: torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Decode a learner action without changing the frozen base template."""
        executed_cpu = _regular_float_tensor(
            executed_normalized_action,
            device="cpu",
        )
        return self.model.decode_policy_decorator_action(
            proposal.model_payload,
            executed_cpu,
        )
