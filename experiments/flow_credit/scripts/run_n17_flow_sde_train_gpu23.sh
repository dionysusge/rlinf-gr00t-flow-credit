#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"

CONFIG_NAME=libero_spatial_ppo_gr00t_n1d7_h200_gpu23_wandb

source "$PROJECT/scripts/activate_rlinf.sh"

export EMBODIED_PATH="$RLINF/examples/embodiment"
export REPO_PATH="$RLINF"
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE=online
export WANDB_ENTITY=liwuyu-cloudbutterfly
export WANDB_PROJECT=GR00T-RL
export WANDB_RUN_GROUP=Flow-SDE-PPO-GR00T-N1.7-LIBERO-Spatial
export WANDB_DIR="$BULK/wandb"
export WANDB_NOTES="GR00T N1.7 LIBERO-Spatial Flow-SDE PPO, physical H200 GPUs 2 and 3, 16 parallel training environments"

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="$BULK/tmp/ray"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2

export PYTHONPATH="$RLINF:${PYTHONPATH:-}"

mkdir -p \
    "$BULK/tmp/ray" \
    "$BULK/runs" \
    "$BULK/logs" \
    "$BULK/wandb"

# 物理卡由 RLinf component_placement=2-3 管理。
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="n17_flow_sde_ppo_gpu23_fp32master_${STAMP}"
RUN_NAME="Flow-SDE-PPO-GR00T-N1.7-LIBERO-Spatial-H200x2-FP32Master-${STAMP}"
RUN_DIR="$BULK/runs/$RUN_ID"

mkdir -p "$RUN_DIR"

echo "$RUN_DIR" \
  > "$BULK/logs/n17_flow_sde_train_gpu23.latest"

cp \
  "$RLINF/examples/embodiment/config/${CONFIG_NAME}.yaml" \
  "$RUN_DIR/launch_config.yaml"

echo "============================================================"
echo "GR00T N1.7 Flow-SDE PPO formal training"
echo "time:             $(date)"
echo "physical GPUs:    2,3"
echo "train envs:       16"
echo "eval envs:        10"
echo "global batch:     64"
echo "micro batch:      4"
echo "offload:          disabled"
echo "grad checkpoint:  disabled"
echo "actor master dtype: FP32"
echo "FSDP compute dtype: BF16"
echo "rollout dtype:      BF16"
echo "W&B project:      liwuyu-cloudbutterfly/GR00T-RL"
echo "W&B run:          $RUN_NAME"
echo "run dir:          $RUN_DIR"
echo "============================================================"

cd "$RLINF"

python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$RUN_DIR" \
    runner.logger.experiment_name="$RUN_NAME"

echo
echo "N17_FLOW_SDE_FORMAL_TRAINING_COMPLETE"
