import json
import os
from pathlib import Path

import hydra
import numpy as np
import ray
import torch
import torch.multiprocessing as mp
from omegaconf import open_dict

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


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_spatial_n17_fixed100_eval_gpu23",
)
def main(cfg) -> None:
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

        metrics = runner.evaluate()
        metrics = {
            key: to_jsonable(value)
            for key, value in metrics.items()
        }

        payload = {
            "label": label,
            "resume_dir": cfg.runner.resume_dir,
            "expected_trajectories": 100,
            "metrics": metrics,
        }

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

        if actual != 100:
            raise RuntimeError(
                f"Expected 100 trajectories, got {actual}"
            )

        print()
        print("FIXED100_EVAL_COMPLETE")
    finally:
        try:
            runner._finish_run()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
