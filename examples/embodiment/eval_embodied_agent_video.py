import json

import hydra
import ray
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_spatial_n17_preppo_video_gpu23",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)

    print("===== resolved evaluation config =====")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

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

        # Actor 从原始 SFT model_path 初始化；
        # resume_dir=None，因此不会载入 PPO checkpoint。
        runner.update_rollout_weights()

        metrics = runner.evaluate()
        prefixed = {f"eval/{key}": value for key, value in metrics.items()}

        print()
        print("===== evaluation metrics =====")
        print(json.dumps(prefixed, indent=2, default=str))

        runner.metric_logger.log(prefixed, step=0)

        print()
        print("PREPPO_VIDEO_EVAL_COMPLETE")
    finally:
        try:
            runner._finish_run()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
