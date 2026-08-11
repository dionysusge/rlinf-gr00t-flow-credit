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

"""Small networks for action-conditioned residual SAC."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    """Build the ReLU MLP used by the actor and critics."""
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be at least 1")
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(width, hidden_dim), nn.ReLU()))
        width = hidden_dim
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ResidualActorOutput:
    """Tanh-squashed residual policy output in unit residual space."""

    action: torch.Tensor
    log_prob: torch.Tensor
    mean_action: torch.Tensor
    pre_tanh_action: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor


class ResidualActor(nn.Module):
    """Action-conditioned tanh-Gaussian actor.

    The network predicts a unit residual in ``[-1, 1]``. The environment glue
    applies the physical residual budget separately as
    ``base_action + residual_scale * unit_residual``. Keeping the scale outside
    this module makes replay contents and SAC log probabilities unambiguous.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        action_shape: tuple[int, int],
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        initial_log_std: float = -2.5,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError("context_dim must be positive")
        if len(action_shape) != 2 or min(action_shape) < 1:
            raise ValueError("action_shape must be (horizon, action_dim)")
        if not min_log_std < max_log_std:
            raise ValueError("min_log_std must be smaller than max_log_std")

        self.context_dim = int(context_dim)
        self.action_shape = (int(action_shape[0]), int(action_shape[1]))
        self.flat_action_dim = math.prod(self.action_shape)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.context_norm = nn.LayerNorm(self.context_dim)
        self.trunk = _mlp(
            self.context_dim + self.flat_action_dim,
            hidden_dim,
            hidden_layers,
        )
        self.mean_head = nn.Linear(hidden_dim, self.flat_action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.flat_action_dim)

        # The deterministic decorator initially reproduces the frozen base
        # policy exactly. Stochastic exploration is small but non-zero.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(initial_log_std))

    def _validate_inputs(
        self, context: torch.Tensor, base_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, {self.context_dim}], "
                f"got {tuple(context.shape)}"
            )
        if base_action.ndim != 3 or tuple(base_action.shape[-2:]) != self.action_shape:
            raise ValueError(
                "base_action must have trailing shape "
                f"{self.action_shape}, got {tuple(base_action.shape)}"
            )
        if context.shape[0] != base_action.shape[0]:
            raise ValueError("context and base_action batch sizes must match")
        if not torch.isfinite(context).all():
            raise FloatingPointError("Non-finite policy context")
        if not torch.isfinite(base_action).all():
            raise FloatingPointError("Non-finite base action")
        return context.float(), base_action.float()

    def distribution_parameters(
        self, context: torch.Tensor, base_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-tanh Gaussian mean and clamped log standard deviation."""
        context, base_action = self._validate_inputs(context, base_action)
        actor_input = torch.cat(
            (self.context_norm(context), base_action.flatten(start_dim=1)), dim=-1
        )
        hidden = self.trunk(actor_input)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        # Algebraically stable tanh Jacobian from the SAC appendix.
        correction = 2.0 * (
            math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action)
        )
        return (distribution.log_prob(pre_tanh_action) - correction).sum(
            dim=-1, keepdim=True
        )

    def sample(
        self,
        context: torch.Tensor,
        base_action: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> ResidualActorOutput:
        """Sample a unit residual and its squashed log probability."""
        mean, log_std = self.distribution_parameters(context, base_action)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh_action)
        mean_action = torch.tanh(mean)
        log_prob = self._squashed_log_prob(distribution, pre_tanh_action)
        batch_size = context.shape[0]
        return ResidualActorOutput(
            action=action.reshape(batch_size, *self.action_shape),
            log_prob=log_prob,
            mean_action=mean_action.reshape(batch_size, *self.action_shape),
            pre_tanh_action=pre_tanh_action.reshape(batch_size, *self.action_shape),
            mean=mean.reshape(batch_size, *self.action_shape),
            log_std=log_std.reshape(batch_size, *self.action_shape),
        )


class SoftQNetwork(nn.Module):
    """One Q network over context and executed normalized action chunks."""

    def __init__(
        self,
        *,
        context_dim: int,
        action_shape: tuple[int, int],
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.action_shape = (int(action_shape[0]), int(action_shape[1]))
        self.flat_action_dim = math.prod(self.action_shape)
        self.context_norm = nn.LayerNorm(self.context_dim)
        self.trunk = _mlp(
            self.context_dim + self.flat_action_dim,
            hidden_dim,
            hidden_layers,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self, context: torch.Tensor, executed_action: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the action chunk actually sent through the GR00T decoder."""
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("Invalid critic context shape")
        if (
            executed_action.ndim != 3
            or tuple(executed_action.shape[-2:]) != self.action_shape
        ):
            raise ValueError("Invalid executed_action shape")
        critic_input = torch.cat(
            (
                self.context_norm(context.float()),
                executed_action.float().flatten(start_dim=1),
            ),
            dim=-1,
        )
        return self.output(self.trunk(critic_input))


class TwinQCritic(nn.Module):
    """Independent twin critics used to reduce positive value bias."""

    def __init__(
        self,
        *,
        context_dim: int,
        action_shape: tuple[int, int],
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        kwargs = {
            "context_dim": context_dim,
            "action_shape": action_shape,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
        }
        self.q1 = SoftQNetwork(**kwargs)
        self.q2 = SoftQNetwork(**kwargs)

    def forward(
        self, context: torch.Tensor, executed_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q estimates."""
        return self.q1(context, executed_action), self.q2(context, executed_action)
