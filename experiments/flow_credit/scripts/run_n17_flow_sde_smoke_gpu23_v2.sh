#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"

CONFIG_NAME=libero_spatial_ppo_gr00t_n1d7_h200_smoke_gpu23_v2

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
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="$BULK/tmp/ray"

# W&B 在线记录。
export WANDB_MODE=online
export WANDB_ENTITY=liwuyu-cloudbutterfly
export WANDB_PROJECT=GR00T-RL
export WANDB_DIR="$BULK/wandb"

export PYTHONPATH="$RLINF:${PYTHONPATH:-}"

mkdir -p \
    "$BULK/tmp/ray" \
    "$BULK/runs" \
    "$BULK/logs" \
    "$BULK/wandb"

# 使用配置中的物理 GPU 2、3。
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true

RUN_ID="n17_flow_sde_smoke_gpu23_v2_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BULK/runs/$RUN_ID"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$BULK/logs/n17_flow_sde_smoke_gpu23_v2.latest"

echo "============================================================"
echo "GR00T N1.7 Flow-SDE PPO smoke v2"
echo "time:       $(date)"
echo "config:     $CONFIG_NAME"
echo "GPU:        physical 2,3"
echo "W&B:        liwuyu-cloudbutterfly/GR00T-RL"
echo "run dir:    $RUN_DIR"
echo "============================================================"

cd "$RLINF"

python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$RUN_DIR"

echo
echo "N17_FLOW_SDE_SMOKE_V2_COMPLETE"
