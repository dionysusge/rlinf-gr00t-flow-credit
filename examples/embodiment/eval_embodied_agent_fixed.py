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

import csv
import json
import os
import signal
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import ray
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import (
    MultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)


def close_worker_groups(groups: tuple[tuple[str, Any], ...]) -> list[str]:
    """Release Ray worker groups in dependency order without masking failures."""
    errors = []
    for name, group in groups:
        if group is None:
            continue
        try:
            group._close()
        except Exception as error:  # pragma: no cover - requires live Ray actors
            errors.append(f"{name}: {type(error).__name__}: {error}")
    return errors


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()

        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def build_trial_records(metric_shards, *, label, checkpoint):
    """Convert aligned per-rank metric tensors into structured trial rows."""
    required = {
        "task_id",
        "trial_id",
        "reset_id",
        "success_once",
        "return",
        "reward",
        "episode_len",
    }
    records = []
    for shard in metric_shards:
        missing = required.difference(shard)
        if missing:
            raise KeyError(f"Evaluation shard is missing fields: {sorted(missing)}")
        lengths = {key: int(shard[key].reshape(-1).shape[0]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Evaluation shard fields are misaligned: {lengths}")
        flattened = {key: shard[key].reshape(-1).tolist() for key in required}
        for index in range(next(iter(lengths.values()))):
            records.append(
                {
                    "task_id": int(flattened["task_id"][index]),
                    "trial_id": int(flattened["trial_id"][index]),
                    "reset_id": int(flattened["reset_id"][index]),
                    "success": int(bool(flattened["success_once"][index])),
                    "return": float(flattened["return"][index]),
                    "reward": float(flattened["reward"][index]),
                    "episode_length": int(flattened["episode_len"][index]),
                    "model": label,
                    "checkpoint": checkpoint,
                }
            )
    records.sort(key=lambda row: (row["task_id"], row["trial_id"]))
    keys = [(row["task_id"], row["trial_id"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Evaluation produced duplicate (task_id, trial_id) records")
    return records


def write_trial_records(output_dir, records):
    """Write trial-aligned CSV and JSONL artifacts."""
    fieldnames = list(records[0]) if records else []
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_spatial_n17_fixed100_eval_gpu23",
)
def main(cfg) -> None:
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("fixed evaluation received SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    cfg = validate_cfg(cfg)

    # The config is validated as evaluation-only, but PPO checkpoints are
    # FSDP actor checkpoints. Enable actor-to-rollout weight synchronization
    # after validation, without creating or running training environments.
    with open_dict(cfg):
        cfg.runner.only_eval = False
        cfg.env.train = None

    assert cfg.weight_syncer is not None
    assert cfg.env.train is None

    output_dir = Path(cfg.runner.logger.log_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)

    label = os.environ.get("EVAL_LABEL", "unknown")

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )

    placements = HybridComponentPlacement(cfg, cluster)

    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placements.get_strategy("actor"),
    )

    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placements.get_strategy("rollout"),
    )

    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placements.get_strategy("env"),
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=None,
    )

    try:
        runner.init_workers()

        # baseline: 从原始 SFT 权重初始化
        # PPO checkpoint: 从 resume_dir/actor 加载
        runner.update_rollout_weights()

        metrics, metric_shards = runner.evaluate(return_metric_shards=True)
        metrics = {key: to_jsonable(value) for key, value in metrics.items()}

        payload = {
            "label": label,
            "resume_dir": cfg.runner.resume_dir,
            "expected_trajectories": int(cfg.env.eval.total_num_envs),
            "eval_reset_offset": int(cfg.env.eval.get("eval_reset_offset", 0)),
            "eval_reset_limit": int(cfg.env.eval.get("eval_reset_limit", 0)),
            "metrics": metrics,
        }

        checkpoint = cfg.runner.resume_dir or "sft_base"
        trial_records = build_trial_records(
            metric_shards,
            label=label,
            checkpoint=checkpoint,
        )
        write_trial_records(output_dir, trial_records)

        metrics_path = output_dir / "metrics.json"

        metrics_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("===== fixed evaluation result =====")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        actual = int(metrics.get("num_trajectories", -1))
        expected = int(cfg.env.eval.total_num_envs)

        if actual != expected:
            raise RuntimeError(f"Expected {expected} trajectories, got {actual}")

        print()
        print("FIXED100_EVAL_COMPLETE")
    finally:
        try:
            try:
                runner._finish_run()
            except Exception as error:
                print(f"Warning: metric logger cleanup failed: {error}")
            cleanup_errors = close_worker_groups(
                (
                    ("env", env_group),
                    ("rollout", rollout_group),
                    ("actor", actor_group),
                )
            )
            for cleanup_error in cleanup_errors:
                print(f"Warning: worker cleanup failed: {cleanup_error}")
        finally:
            try:
                ray.shutdown()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    main()
