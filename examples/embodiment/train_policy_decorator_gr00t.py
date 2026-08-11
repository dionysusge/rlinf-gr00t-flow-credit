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

"""Run frozen-GR00T Policy Decorator smoke gates and standalone SAC."""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.policy_decorator import (
    DecoratorReplayBuffer,
    GR00TPolicyDecoratorAdapter,
    PolicyDecoratorSAC,
    PolicyDecoratorSACConfig,
    ProgressiveResidualSchedule,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models import get_model
from rlinf.utils.metric_logger import MetricLogger

VALID_MODES = {"zero_smoke", "random_smoke", "train", "eval"}


def _seed_everything(seed: int) -> None:
    """Seed model, learner, and host RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    """Convert scalar metric values to JSON-compatible Python objects."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


class PolicyDecoratorExperiment:
    """Single-process glue runner with GR00T and SAC on separate GPUs."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.pd_cfg = cfg.policy_decorator
        self.mode = str(self.pd_cfg.mode)
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")

        self.model_device = torch.device(str(self.pd_cfg.model_device))
        self.learner_device = torch.device(str(self.pd_cfg.learner_device))
        self.seed = int(self.pd_cfg.seed)
        _seed_everything(self.seed)

        self.run_root = Path(str(cfg.runner.logger.log_path)).expanduser().resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.gate_dir = self.run_root / "gates"
        self.checkpoint_dir = self.run_root / "checkpoints"
        self.gate_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_root / f"metrics_{self.mode}.jsonl"
        self.summary_path = self.run_root / f"summary_{self.mode}.json"
        OmegaConf.save(cfg, self.run_root / f"config_{self.mode}.yaml", resolve=True)
        self._check_training_gates()

        self.metric_logger = MetricLogger(cfg)
        self.model = get_model(cfg.actor.model)
        if self.model is None:
            raise RuntimeError("Could not construct configured GR00T model")
        self.model.to(self.model_device)
        self.model.eval()
        # Model construction/loading consumes RNG. Restore the experiment seed
        # so seed=N always identifies the same first GR00T flow latent.
        _seed_everything(self.seed)
        self.adapter = GR00TPolicyDecoratorAdapter(
            self.model,
            learner_device=self.learner_device,
        )

        env_cfg = cfg.env.train if self.mode == "train" else cfg.env.eval
        self.env_cfg = env_cfg
        self.num_envs = int(env_cfg.total_num_envs)
        self.env = LiberoEnv(
            env_cfg,
            num_envs=self.num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        # Environment construction may initialize libraries that touch the host
        # RNGs. No stochastic policy call is allowed before this final reseed.
        _seed_everything(self.seed)

        self.learner: PolicyDecoratorSAC | None = None
        self.replay: DecoratorReplayBuffer | None = None
        self.decision_step = 0
        self.env_transitions = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.nonfinite_count = 0
        self.zero_identity_checks = 0
        self.failure_reason: str | None = None
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self.recent_success = deque(maxlen=int(self.pd_cfg.success_window))
        self.recent_return = deque(maxlen=int(self.pd_cfg.success_window))
        self.recent_length = deque(maxlen=int(self.pd_cfg.success_window))

    def _build_learner(self, context_dim: int) -> None:
        expected_context_dim = self.pd_cfg.get("expected_context_dim", None)
        if (
            expected_context_dim is not None
            and int(expected_context_dim) != context_dim
        ):
            raise ValueError(
                f"GR00T context width is {context_dim}, configured expectation is "
                f"{expected_context_dim}"
            )
        sac_config = PolicyDecoratorSACConfig(
            context_dim=context_dim,
            action_horizon=int(self.pd_cfg.action_horizon),
            action_dim=int(self.pd_cfg.action_dim),
            residual_scale=float(self.pd_cfg.residual_scale),
            hidden_dim=int(self.pd_cfg.hidden_dim),
            hidden_layers=int(self.pd_cfg.hidden_layers),
            actor_lr=float(self.pd_cfg.actor_lr),
            critic_lr=float(self.pd_cfg.critic_lr),
            alpha_lr=float(self.pd_cfg.alpha_lr),
            discount=float(self.pd_cfg.discount),
            target_tau=float(self.pd_cfg.target_tau),
            initial_alpha=float(self.pd_cfg.initial_alpha),
            target_entropy=self.pd_cfg.get("target_entropy", None),
            initial_log_std=float(self.pd_cfg.initial_log_std),
            min_log_std=float(self.pd_cfg.min_log_std),
            max_log_std=float(self.pd_cfg.max_log_std),
            max_grad_norm=float(self.pd_cfg.max_grad_norm),
            clip_executed_action=bool(self.pd_cfg.clip_executed_action),
        )
        self.learner = PolicyDecoratorSAC(
            sac_config,
            device=self.learner_device,
        )
        storage_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[str(self.pd_cfg.replay_storage_dtype)]
        self.replay = DecoratorReplayBuffer(
            capacity=int(self.pd_cfg.replay_capacity),
            context_dim=context_dim,
            action_shape=(int(self.pd_cfg.action_horizon), int(self.pd_cfg.action_dim)),
            storage_dtype=storage_dtype,
            seed=self.seed + 17,
        )

    def _check_training_gates(self) -> None:
        if self.mode != "train" or bool(self.pd_cfg.skip_gate_check):
            return
        required = ("zero_smoke", "random_smoke")
        failures = []
        for gate_name in required:
            path = self.gate_dir / f"{gate_name}.json"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            if not bool(result.get("passed", False)):
                failures.append(f"failed {path}")
        if failures:
            raise RuntimeError(
                "Refusing to train before both smoke gates pass: " + ", ".join(failures)
            )

    def _select_residual(
        self, context: torch.Tensor, base_action: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.learner is not None
        action_shape = base_action.shape
        if self.mode == "zero_smoke":
            unit_residual = torch.zeros(action_shape, device=self.learner_device)
            active_ratio = 0.0
        elif self.mode == "random_smoke":
            unit_residual = torch.randn(action_shape, device=self.learner_device)
            unit_residual = (
                unit_residual * float(self.pd_cfg.random_residual_std)
            ).clamp(-1.0, 1.0)
            active_ratio = 1.0
        elif self.mode == "eval":
            policy = self.learner.act(
                context,
                base_action,
                deterministic=True,
            )
            unit_residual = policy.action
            active_ratio = 1.0
        else:
            assert self.replay is not None
            if len(self.replay) < int(self.pd_cfg.learning_starts):
                # Match upstream Policy Decorator: populate early replay with
                # uniform residual actions, not samples from an untrained actor.
                unit_residual = torch.empty_like(base_action).uniform_(-1.0, 1.0)
                prelearning_uniform = 1.0
            else:
                policy = self.learner.act(
                    context,
                    base_action,
                    deterministic=False,
                )
                unit_residual = policy.action
                prelearning_uniform = 0.0
            schedule = ProgressiveResidualSchedule(
                int(self.pd_cfg.progressive_exploration_steps)
            )
            mask = schedule.mask(
                step=self.decision_step,
                batch_size=base_action.shape[0],
                device=self.learner_device,
            )
            unit_residual = torch.where(
                mask,
                unit_residual,
                torch.zeros_like(unit_residual),
            )
            active_ratio = float(mask.float().mean())

        scaled = float(self.pd_cfg.residual_scale) * unit_residual
        step_l2 = torch.linalg.vector_norm(scaled.float(), dim=-1)
        metrics = {
            "residual/l2_mean": float(step_l2.mean()),
            "residual/l2_max": float(step_l2.max()),
            "residual/abs_mean": float(scaled.abs().mean()),
            "residual/unit_saturation_fraction": float(
                (unit_residual.abs() > 0.95).float().mean()
            ),
            "residual/active_env_fraction": active_ratio,
            "residual/prelearning_uniform": (
                prelearning_uniform if self.mode == "train" else 0.0
            ),
        }
        return unit_residual, metrics

    def _record_episodes(
        self,
        chunk_rewards: torch.Tensor,
        chunk_terminations: torch.Tensor,
        chunk_truncations: torch.Tensor,
        infos_list: list[dict[str, Any]],
    ) -> dict[str, float]:
        rewards = chunk_rewards.detach().cpu().float()
        terminated = chunk_terminations.detach().cpu().bool().any(dim=1)
        truncated = chunk_truncations.detach().cpu().bool().any(dim=1)
        done = terminated | truncated
        previous_lengths = self.episode_lengths.clone()
        self.episode_returns += rewards.sum(dim=1)
        self.episode_lengths += rewards.shape[1]

        def episode_info(info: dict[str, Any]) -> dict[str, Any] | None:
            final_info = info.get("final_info")
            if isinstance(final_info, dict) and isinstance(
                final_info.get("episode"), dict
            ):
                return final_info["episode"]
            value = info.get("episode")
            return value if isinstance(value, dict) else None

        def scalar(value: Any, index: int) -> float:
            return float(torch.as_tensor(value)[index].detach().cpu())

        for index in done.nonzero(as_tuple=False).flatten().tolist():
            success = bool(terminated[index])
            episode_return = float(self.episode_returns[index])
            episode_length = int(self.episode_lengths[index])
            final_episode_info = episode_info(infos_list[-1])
            if final_episode_info is not None:
                if "return" in final_episode_info:
                    episode_return = scalar(final_episode_info["return"], index)
                if "episode_len" in final_episode_info:
                    episode_length = int(
                        scalar(final_episode_info["episode_len"], index)
                    )
            if success:
                # chunk_step deliberately postpones auto-reset until the chunk
                # ends. Find the first success flag so episode length is not
                # rounded up to a multiple of the 16-step action horizon.
                for inner_step, info in enumerate(infos_list):
                    current_episode_info = episode_info(info)
                    if current_episode_info is None:
                        continue
                    success_once = current_episode_info.get("success_once")
                    if success_once is not None and bool(scalar(success_once, index)):
                        episode_length = int(previous_lengths[index]) + inner_step + 1
                        break
            self.completed_episodes += 1
            self.successful_episodes += int(success)
            self.recent_success.append(float(success))
            self.recent_return.append(episode_return)
            self.recent_length.append(float(episode_length))
            self.episode_returns[index] = 0
            self.episode_lengths[index] = 0
        return {
            "env/completed_episodes": float(self.completed_episodes),
            "env/success_rate_total": (
                float(self.successful_episodes / self.completed_episodes)
                if self.completed_episodes
                else 0.0
            ),
            "env/success_rate_window": (
                float(np.mean(self.recent_success)) if self.recent_success else 0.0
            ),
            "env/episode_return_window": (
                float(np.mean(self.recent_return)) if self.recent_return else 0.0
            ),
            "env/episode_length_window": (
                float(np.mean(self.recent_length)) if self.recent_length else 0.0
            ),
        }

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        record = {
            "decision_step": self.decision_step,
            "env_transitions": self.env_transitions,
            "wall_time": time.time(),
            "mode": self.mode,
            **{key: _jsonable(value) for key, value in metrics.items()},
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        scalar_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.floating, np.integer))
        }
        self.metric_logger.log(scalar_metrics, step=self.decision_step)

    def _save_checkpoint(self, *, suffix: str | None = None) -> Path:
        assert self.learner is not None and self.replay is not None
        label = suffix or f"step_{self.decision_step:06d}"
        path = self.checkpoint_dir / f"policy_decorator_{label}.pt"
        temporary_path = path.with_suffix(".tmp")
        state = {
            "decision_step": self.decision_step,
            "env_transitions": self.env_transitions,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "learner": self.learner.state_dict(),
            "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
        return path

    def _load_checkpoint(self, path: str) -> None:
        assert self.learner is not None and self.replay is not None
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.learner.load_state_dict(state["learner"])
        self.replay.load_state_dict(state["replay"])
        self.decision_step = int(state["decision_step"])
        self.env_transitions = int(state["env_transitions"])
        self.completed_episodes = int(state["completed_episodes"])
        self.successful_episodes = int(state["successful_episodes"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])

    def _should_stop(self) -> bool:
        if self.mode in ("zero_smoke", "random_smoke", "eval"):
            episode_limit = int(self.pd_cfg[f"{self.mode}_episodes"])
            return self.completed_episodes >= episode_limit
        return self.decision_step >= int(self.pd_cfg.max_steps)

    def _write_summary(self) -> dict[str, Any]:
        success_rate = (
            self.successful_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        summary = {
            "mode": self.mode,
            "decision_steps": self.decision_step,
            "env_transitions": self.env_transitions,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": success_rate,
            "nonfinite_count": self.nonfinite_count,
            "zero_identity_checks": self.zero_identity_checks,
            "failure_reason": self.failure_reason,
            "metrics_path": str(self.metrics_path),
        }
        if self.mode == "zero_smoke":
            minimum_success_rate = self.pd_cfg.get("zero_smoke_min_success_rate", None)
            success_gate_passed = minimum_success_rate is None or success_rate >= float(
                minimum_success_rate
            )
            summary["success_rate_gate_enabled"] = minimum_success_rate is not None
            summary["success_rate_gate_passed"] = success_gate_passed
            summary["passed"] = (
                self.completed_episodes >= int(self.pd_cfg.zero_smoke_episodes)
                and self.zero_identity_checks > 0
                and success_gate_passed
                and self.nonfinite_count == 0
                and self.failure_reason is None
            )
        elif self.mode == "random_smoke":
            summary["passed"] = (
                self.completed_episodes >= int(self.pd_cfg.random_smoke_episodes)
                and self.successful_episodes
                >= int(self.pd_cfg.random_smoke_min_successes)
                and self.nonfinite_count == 0
                and self.failure_reason is None
            )
        else:
            summary["passed"] = (
                self.nonfinite_count == 0 and self.failure_reason is None
            )
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        if self.mode in ("zero_smoke", "random_smoke"):
            gate_path = self.gate_dir / f"{self.mode}.json"
            with gate_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        return summary

    def run(self) -> dict[str, Any]:
        """Collect chunks, update SAC when requested, and persist evidence."""
        obs, _ = self.env.reset()
        proposal = self.adapter.propose(obs)
        self._build_learner(int(proposal.context.shape[-1]))
        assert self.learner is not None and self.replay is not None
        if tuple(proposal.base_action.shape[-2:]) != (
            int(self.pd_cfg.action_horizon),
            int(self.pd_cfg.action_dim),
        ):
            raise ValueError(
                "GR00T proposal action shape does not match Policy Decorator config"
            )
        resume_path = self.pd_cfg.get("resume_path", None)
        if resume_path:
            self._load_checkpoint(str(resume_path))
            # The simulator itself restarts on resume. Rebuild the first new
            # proposal only after restoring the model-side RNG state.
            proposal = self.adapter.propose(obs)

        try:
            while not self._should_stop():
                unit_residual, residual_metrics = self._select_residual(
                    proposal.context,
                    proposal.base_action,
                )
                executed = self.learner.compose_action(
                    proposal.base_action,
                    unit_residual,
                )
                if self.mode == "zero_smoke" and not torch.equal(
                    executed, proposal.base_action
                ):
                    raise AssertionError(
                        "zero_smoke must reproduce the normalized GR00T base action "
                        "bit-for-bit"
                    )
                if self.mode == "zero_smoke":
                    self.zero_identity_checks += int(executed.shape[0])
                raw_action = self.adapter.decode(proposal, executed)
                env_action = prepare_actions(
                    raw_chunk_actions=raw_action,
                    env_type=str(self.env_cfg.env_type),
                    model_type=str(self.cfg.actor.model.model_type),
                    num_action_chunks=int(self.cfg.actor.model.num_action_chunks),
                    action_dim=int(self.cfg.actor.model.action_dim),
                )
                if not torch.isfinite(torch.as_tensor(env_action).float()).all():
                    raise FloatingPointError(
                        "Non-finite LIBERO action after conversion"
                    )

                (
                    obs_list,
                    chunk_rewards,
                    chunk_terminations,
                    chunk_truncations,
                    infos_list,
                ) = self.env.chunk_step(env_action)
                next_obs = obs_list[-1]
                next_proposal = self.adapter.propose(next_obs)
                done = (
                    chunk_terminations.bool().any(dim=1)
                    | chunk_truncations.bool().any(dim=1)
                ).float()[:, None]
                chunk_reward = chunk_rewards.float().sum(dim=1, keepdim=True)

                self.replay.add_batch(
                    context=proposal.context,
                    base_action=proposal.base_action,
                    residual_action=unit_residual,
                    executed_action=executed,
                    reward=chunk_reward,
                    done=done,
                    next_context=next_proposal.context,
                    next_base_action=next_proposal.base_action,
                )
                self.decision_step += 1
                self.env_transitions += self.num_envs

                metrics = {
                    **residual_metrics,
                    **self._record_episodes(
                        chunk_rewards,
                        chunk_terminations,
                        chunk_truncations,
                        infos_list,
                    ),
                    "rollout/chunk_reward_mean": float(chunk_reward.mean()),
                    "rollout/base_action_abs_mean": float(
                        proposal.base_action.abs().mean()
                    ),
                    "rollout/executed_action_abs_mean": float(executed.abs().mean()),
                    "replay/size": float(len(self.replay)),
                }
                if self.mode == "train" and len(self.replay) >= max(
                    int(self.pd_cfg.learning_starts),
                    int(self.pd_cfg.batch_size),
                ):
                    update_metrics: list[dict[str, float]] = []
                    for _ in range(int(self.pd_cfg.updates_per_step)):
                        batch = self.replay.sample(int(self.pd_cfg.batch_size))
                        update_metrics.append(self.learner.update(batch))
                    for key in update_metrics[0]:
                        metrics[key] = float(
                            np.mean([entry[key] for entry in update_metrics])
                        )
                self._append_metrics(metrics)

                if (
                    self.mode == "train"
                    and int(self.pd_cfg.checkpoint_interval) > 0
                    and self.decision_step % int(self.pd_cfg.checkpoint_interval) == 0
                ):
                    self._save_checkpoint()
                proposal = next_proposal

            if self.mode == "train":
                self._save_checkpoint(suffix="final")
            summary = self._write_summary()
            if not bool(summary["passed"]):
                raise RuntimeError(f"Policy Decorator gate failed: {summary}")
            return summary
        except Exception as error:
            self.failure_reason = f"{type(error).__name__}: {error}"
            if isinstance(error, FloatingPointError):
                self.nonfinite_count += 1
            if self.mode == "train" and self.learner is not None:
                self._save_checkpoint(suffix=f"crash_step_{self.decision_step:06d}")
            self._write_summary()
            raise
        finally:
            self.metric_logger.finish()
            if hasattr(self.env, "env") and hasattr(self.env.env, "close"):
                self.env.env.close()
            else:
                close = getattr(self.env, "close", None)
                if callable(close):
                    close()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_spatial_policy_decorator_gr00t",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    experiment = PolicyDecoratorExperiment(cfg)
    summary = experiment.run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
