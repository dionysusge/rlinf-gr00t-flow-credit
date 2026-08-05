#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"

CONFIG_NAME=libero_spatial_ppo_gr00t_n1d7_h200_smoke_gpu23

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

export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="$BULK/tmp/ray"
export HYDRA_FULL_ERROR=1

export PYTHONPATH="$RLINF:${PYTHONPATH:-}"

mkdir -p \
    "$BULK/tmp/ray" \
    "$BULK/runs" \
    "$BULK/logs"

# 不使用 CUDA_VISIBLE_DEVICES。
# 物理卡 2、3 由 cluster.component_placement 指定。
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true

RUN_ID="n17_flow_sde_smoke_gpu23_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BULK/runs/$RUN_ID"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$BULK/logs/n17_flow_sde_smoke_gpu23.latest"

echo "============================================================"
echo "GR00T-N1.7 Flow-SDE PPO smoke test"
echo "time:        $(date)"
echo "python:      $(command -v python)"
echo "config:      $CONFIG_NAME"
echo "physical GPU placement: 2-3"
echo "run dir:     $RUN_DIR"
echo "============================================================"

cd "$RLINF"

python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$RUN_DIR"

echo
echo "N17_FLOW_SDE_SMOKE_COMPLETE"
