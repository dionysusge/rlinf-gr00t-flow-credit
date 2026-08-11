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

"""Evaluate fixed LIBERO trials with one frozen GR00T process and no Ray."""

from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.flow_credit.analysis.fixed_eval_records import (
    episode_records,
    to_finite_tensor,
    write_trial_records,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models import get_model


def _seed_model_stream(seed: int) -> None:
    """Seed the sampled GR00T flow latent independently from LIBERO resets."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FixedNoRayEvaluator:
    """Own one GR00T model and one local LIBERO vector environment."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.eval_cfg = cfg.env.eval
        self.direct_cfg = cfg.direct_eval
        self.output_dir = Path(str(cfg.runner.logger.log_path)).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            cfg,
            self.output_dir / "resolved_config.yaml",
            resolve=True,
        )
        self.label = str(self.direct_cfg.label)
        self.expected_trials = int(self.direct_cfg.expected_trials)
        self.device = torch.device(str(self.direct_cfg.model_device))

        # Loading checkpoint modules may consume framework RNG. The flow stream
        # is deliberately reset again after model and environment construction.
        _seed_model_stream(int(self.direct_cfg.model_seed))
        self.model = get_model(cfg.actor.model)
        if self.model is None:
            raise RuntimeError("Could not construct the configured GR00T model")
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.env = LiberoEnv(
            self.eval_cfg,
            num_envs=int(self.eval_cfg.total_num_envs),
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )

    def run(self) -> dict[str, Any]:
        """Evaluate until every reset in the configured fixed slice completes."""
        obs, _ = self.env.reset()
        _seed_model_stream(int(self.direct_cfg.model_seed))
        records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        checkpoint = str(self.cfg.runner.get("resume_dir", None) or "sft_base")
        decision_steps = 0
        max_decision_steps = int(self.direct_cfg.max_decision_steps)

        try:
            while len(records_by_key) < self.expected_trials:
                completed_before_step = len(records_by_key)
                raw_action, _ = self.model.predict_action_batch(obs, mode="eval")
                raw_tensor = to_finite_tensor(raw_action)
                if raw_tensor.ndim != 3:
                    raise ValueError(
                        "GR00T evaluation action must be [batch,horizon,dim], "
                        f"got {tuple(raw_tensor.shape)}"
                    )
                env_action = prepare_actions(
                    raw_chunk_actions=raw_action,
                    env_type=str(self.eval_cfg.env_type),
                    model_type=str(self.cfg.actor.model.model_type),
                    num_action_chunks=int(self.cfg.actor.model.num_action_chunks),
                    action_dim=int(self.cfg.actor.model.action_dim),
                )
                if not torch.isfinite(torch.as_tensor(env_action).float()).all():
                    raise FloatingPointError("Non-finite decoded LIBERO action")

                (
                    obs_list,
                    _chunk_rewards,
                    _chunk_terminations,
                    _chunk_truncations,
                    infos_list,
                ) = self.env.chunk_step(env_action)
                obs = obs_list[-1]
                for record in episode_records(
                    infos_list,
                    label=self.label,
                    checkpoint=checkpoint,
                ):
                    key = (record["task_id"], record["trial_id"])
                    if key in records_by_key:
                        raise ValueError(f"Duplicate completed trial {key}")
                    records_by_key[key] = record
                decision_steps += 1
                if len(records_by_key) != completed_before_step:
                    print(
                        f"[{self.label}] completed {len(records_by_key)}/"
                        f"{self.expected_trials} trials after {decision_steps} "
                        "action chunks",
                        flush=True,
                    )
                if (
                    len(records_by_key) < self.expected_trials
                    and decision_steps >= max_decision_steps
                ):
                    raise TimeoutError(
                        f"Collected {len(records_by_key)}/{self.expected_trials} "
                        f"trials after {decision_steps} action chunks"
                    )

            records = list(records_by_key.values())
            if len(records) != self.expected_trials:
                raise RuntimeError(
                    f"Expected {self.expected_trials} trials, got {len(records)}"
                )
            write_trial_records(self.output_dir, records)
            successes = sum(int(record["success"]) for record in records)
            metrics = {
                "num_trajectories": len(records),
                "successes": successes,
                "success_rate": successes / len(records),
                "reward": sum(float(record["reward"]) for record in records)
                / len(records),
                "episode_length": sum(
                    int(record["episode_length"]) for record in records
                )
                / len(records),
                "decision_steps": decision_steps,
                "runtime": "direct_no_ray",
                "ray_initialized": False,
                "model_seed": int(self.direct_cfg.model_seed),
                "parallel_envs": int(self.eval_cfg.total_num_envs),
                "eval_reset_offset": int(self.eval_cfg.eval_reset_offset),
                "eval_reset_limit": int(self.eval_cfg.eval_reset_limit),
            }
            payload = {
                "label": self.label,
                "resume_dir": self.cfg.runner.get("resume_dir", None),
                "expected_trajectories": self.expected_trials,
                "metrics": metrics,
            }
            (self.output_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (self.output_dir / "NO_RAY_EVAL_COMPLETE").write_text(
                f"{successes}/{len(records)} successes\n",
                encoding="utf-8",
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print("FIXED100_EVAL_COMPLETE")
            return payload
        finally:
            try:
                if hasattr(self.env, "env") and hasattr(self.env.env, "close"):
                    self.env.env.close()
            finally:
                try:
                    self.model.to("cpu")
                    del self.model
                finally:
                    if torch.cuda.is_available() and self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                        torch.cuda.empty_cache()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_spatial_n17_fixed100_eval_no_ray_gpu4",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("E0 evaluator received SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        FixedNoRayEvaluator(cfg).run()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
