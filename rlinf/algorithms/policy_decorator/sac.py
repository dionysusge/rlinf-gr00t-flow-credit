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

"""Standalone SAC learner for frozen-policy action decoration."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from .networks import ResidualActor, ResidualActorOutput, TwinQCritic
from .replay_buffer import DecoratorTransitionBatch


@dataclass(frozen=True)
class PolicyDecoratorSACConfig:
    """Hyperparameters for the standalone residual SAC learner."""

    context_dim: int
    action_horizon: int = 16
    action_dim: int = 7
    residual_scale: float = 0.1
    hidden_dim: int = 256
    hidden_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    discount: float = 0.99
    target_tau: float = 0.005
    initial_alpha: float = 0.1
    target_entropy: float | None = None
    initial_log_std: float = -2.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    max_grad_norm: float = 10.0
    clip_executed_action: bool = False

    def __post_init__(self) -> None:
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive")
        if self.action_horizon < 1 or self.action_dim < 1:
            raise ValueError("action dimensions must be positive")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")


def _finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite SAC tensor: {name}")


class PolicyDecoratorSAC:
    """Own the residual actor, twin critics, targets, and optimizers.

    This object is deliberately independent from Hugging Face, FSDP, Ray, and
    RLinf PPO workers. Only ordinary float32 tensors cross its API boundary.
    """

    def __init__(
        self,
        config: PolicyDecoratorSACConfig,
        *,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        action_shape = (config.action_horizon, config.action_dim)
        network_kwargs = {
            "context_dim": config.context_dim,
            "action_shape": action_shape,
            "hidden_dim": config.hidden_dim,
            "hidden_layers": config.hidden_layers,
        }
        self.actor = ResidualActor(
            **network_kwargs,
            initial_log_std=config.initial_log_std,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
        ).to(self.device)
        self.critic = TwinQCritic(**network_kwargs).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = (
            -float(config.action_horizon * config.action_dim)
            if config.target_entropy is None
            else float(config.target_entropy)
        )
        self.update_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Current detached entropy coefficient."""
        return self.log_alpha.exp().detach()

    def compose_action(
        self, base_action: torch.Tensor, unit_residual: torch.Tensor
    ) -> torch.Tensor:
        """Apply the one canonical residual composition rule."""
        base_action = base_action.float()
        unit_residual = unit_residual.float()
        if base_action.shape != unit_residual.shape:
            raise ValueError("base_action and residual shapes must match")
        _finite_or_raise("base_action", base_action)
        _finite_or_raise("unit_residual", unit_residual)
        if (unit_residual.abs() > 1.00001).any():
            raise ValueError("unit_residual must remain inside [-1, 1]")
        action = base_action + self.config.residual_scale * unit_residual
        if self.config.clip_executed_action:
            action = action.clamp(-1.0, 1.0)
        _finite_or_raise("executed_action", action)
        return action

    @torch.no_grad()
    def act(
        self,
        context: torch.Tensor,
        base_action: torch.Tensor,
        *,
        deterministic: bool,
    ) -> ResidualActorOutput:
        """Sample a residual action for collection or evaluation."""
        return self.actor.sample(
            context.to(self.device, dtype=torch.float32),
            base_action.to(self.device, dtype=torch.float32),
            deterministic=deterministic,
        )

    def update(self, batch: DecoratorTransitionBatch) -> dict[str, float]:
        """Perform one twin-Q, policy, temperature, and target update."""
        batch = batch.to(self.device)
        for name, value in batch.__dict__.items():
            _finite_or_raise(name, value)

        with torch.no_grad():
            next_policy = self.actor.sample(
                batch.next_context, batch.next_base_action, deterministic=False
            )
            next_executed = self.compose_action(
                batch.next_base_action, next_policy.action
            )
            target_q1, target_q2 = self.target_critic(batch.next_context, next_executed)
            min_target_q = torch.minimum(target_q1, target_q2)
            target = batch.reward + (1.0 - batch.done) * self.config.discount * (
                min_target_q - self.alpha * next_policy.log_prob
            )
            _finite_or_raise("critic_target", target)

        q1, q2 = self.critic(batch.context, batch.executed_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        _finite_or_raise("critic_loss", critic_loss)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.critic_optimizer.step()

        policy = self.actor.sample(
            batch.context, batch.base_action, deterministic=False
        )
        actor_executed = self.compose_action(batch.base_action, policy.action)
        actor_q1, actor_q2 = self.critic(batch.context, actor_executed)
        min_actor_q = torch.minimum(actor_q1, actor_q2)
        actor_loss = (self.alpha * policy.log_prob - min_actor_q).mean()
        _finite_or_raise("actor_loss", actor_loss)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (policy.log_prob.detach() + self.target_entropy)
        ).mean()
        _finite_or_raise("alpha_loss", alpha_loss)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.target_tau)

        self.update_step += 1
        return {
            "train/critic_loss": float(critic_loss.detach()),
            "train/actor_loss": float(actor_loss.detach()),
            "train/alpha_loss": float(alpha_loss.detach()),
            "train/alpha": float(self.alpha),
            "train/q1_mean": float(q1.detach().mean()),
            "train/q2_mean": float(q2.detach().mean()),
            "train/target_q_mean": float(target.detach().mean()),
            "train/log_prob_mean": float(policy.log_prob.detach().mean()),
            "train/actor_grad_norm": float(actor_grad_norm),
            "train/critic_grad_norm": float(critic_grad_norm),
        }

    def state_dict(self) -> dict[str, object]:
        """Return a complete learner checkpoint."""
        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore learner and optimizer state."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data.copy_(
            torch.as_tensor(state["log_alpha"], device=self.device)
        )
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.update_step = int(state["update_step"])
