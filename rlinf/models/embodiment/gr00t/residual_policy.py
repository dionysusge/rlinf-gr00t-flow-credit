# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small Gaussian residual policy used to refine frozen VLA actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class ResidualPolicyOutput:
    """A sampled residual and the statistics needed by PPO."""

    raw_action: torch.Tensor
    action: torch.Tensor
    logprobs: torch.Tensor
    entropy: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor


class GaussianResidualActor(nn.Module):
    """Predict a bounded Gaussian residual action chunk from frozen VLA features.

    The Gaussian is defined in an unconstrained raw space. The environment-facing
    normalized correction is ``bound * scale * tanh(raw_action)``. PPO evaluates
    the recorded raw sample under the current Gaussian. The tanh Jacobian and the
    fixed scale cancel in the old/new probability ratio.
    """

    def __init__(
        self,
        *,
        vlm_width: int,
        state_width: int,
        hidden_width: int,
        hidden_layers: int,
        action_horizon: int,
        action_dim: int,
        residual_bound: float,
        initial_log_std: float = -2.5,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if action_horizon < 1 or action_dim < 1:
            raise ValueError("action_horizon and action_dim must be positive")
        if residual_bound < 0:
            raise ValueError("residual_bound must be non-negative")

        self.vlm_width = int(vlm_width)
        self.state_width = int(state_width)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.residual_bound = float(residual_bound)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        input_width = self.vlm_width + self.state_width
        layers: list[nn.Module] = [
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
        ]
        for _ in range(hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_width, hidden_width),
                    nn.SiLU(),
                ]
            )
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(
            hidden_width,
            self.action_horizon * self.action_dim,
        )
        self.log_std = nn.Parameter(
            torch.full(
                (self.action_horizon, self.action_dim),
                float(initial_log_std),
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Make the initial mean correction exactly zero."""
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def _features(
        self,
        vlm_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        if vlm_features.ndim != 3:
            raise ValueError(
                "vlm_features must have shape [batch, tokens, width], got "
                f"{tuple(vlm_features.shape)}"
            )
        if state_features.ndim < 2:
            raise ValueError(
                "state_features must have a batch dimension, got "
                f"{tuple(state_features.shape)}"
            )
        vlm_pooled = vlm_features.mean(dim=1)
        state_flat = state_features.reshape(state_features.shape[0], -1)
        if vlm_pooled.shape[-1] != self.vlm_width:
            raise ValueError(
                f"Expected VLM width {self.vlm_width}, got {vlm_pooled.shape[-1]}"
            )
        if state_flat.shape[-1] != self.state_width:
            raise ValueError(
                f"Expected state width {self.state_width}, got {state_flat.shape[-1]}"
            )
        return torch.cat((vlm_pooled, state_flat), dim=-1)

    def distribution(
        self,
        vlm_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        """Return the raw-space Gaussian, mean and clamped log standard deviation."""
        features = self._features(vlm_features, state_features)
        hidden = self.trunk(features)
        mean = self.mean_head(hidden).reshape(
            features.shape[0],
            self.action_horizon,
            self.action_dim,
        )
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        log_std = log_std.unsqueeze(0).expand_as(mean)
        return Normal(mean, log_std.exp()), mean, log_std

    def sample(
        self,
        vlm_features: torch.Tensor,
        state_features: torch.Tensor,
        *,
        deterministic: bool,
        force_zero: bool = False,
        scale: float = 1.0,
    ) -> ResidualPolicyOutput:
        """Sample a raw residual and map it into bounded normalized action space."""
        distribution, mean, log_std = self.distribution(vlm_features, state_features)
        if force_zero:
            raw_action = torch.zeros_like(mean)
            action = torch.zeros_like(mean)
        else:
            raw_action = mean if deterministic else distribution.sample()
            action = self.residual_bound * float(scale) * torch.tanh(raw_action)
        return ResidualPolicyOutput(
            raw_action=raw_action,
            action=action,
            logprobs=distribution.log_prob(raw_action),
            entropy=distribution.entropy(),
            mean=mean,
            log_std=log_std,
        )

    def evaluate_actions(
        self,
        vlm_features: torch.Tensor,
        state_features: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate recorded raw samples for PPO training."""
        distribution, _, _ = self.distribution(vlm_features, state_features)
        return distribution.log_prob(raw_actions), distribution.entropy()


def summarize_residual_actions(
    residual_actions: torch.Tensor,
    *,
    dimension_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Create scalar diagnostics for bounded normalized residual chunks."""
    if residual_actions.ndim < 3:
        raise ValueError(
            "residual_actions must end in [horizon, action_dim], got "
            f"{tuple(residual_actions.shape)}"
        )
    actions = residual_actions.detach().float()
    horizon = actions.shape[-2]
    action_dim = actions.shape[-1]
    flat = actions.reshape(-1, horizon, action_dim)
    step_norm = torch.linalg.vector_norm(flat, dim=-1)

    metrics = {
        "residual/l2_mean": float(step_norm.mean().item()),
        "residual/l2_max": float(step_norm.max().item()),
        "residual/l2_p95": float(torch.quantile(step_norm, 0.95).item()),
        "residual/abs_mean": float(flat.abs().mean().item()),
        "residual/active_fraction_gt_0.01": float((step_norm > 0.01).float().mean()),
        "residual/active_fraction_gt_0.05": float((step_norm > 0.05).float().mean()),
    }

    names = dimension_names or tuple(f"dim_{idx}" for idx in range(action_dim))
    if len(names) != action_dim:
        raise ValueError(
            f"Expected {action_dim} dimension names, received {len(names)}"
        )
    for idx, name in enumerate(names):
        metrics[f"residual/dimension/{name}_abs_mean"] = float(
            flat[..., idx].abs().mean().item()
        )
    for idx in range(horizon):
        metrics[f"residual/horizon/{idx:02d}_l2_mean"] = float(
            step_norm[:, idx].mean().item()
        )
    return metrics
