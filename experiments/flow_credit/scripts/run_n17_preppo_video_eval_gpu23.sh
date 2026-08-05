#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"

CONFIG_NAME=libero_spatial_n17_preppo_video_gpu23

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

export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="$BULK/tmp/ray"
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"

mkdir -p \
    "$BULK/tmp/ray" \
    "$BULK/evaluations" \
    "$BULK/logs"

unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true

RUN_ID="n17_libero_spatial_preppo_video_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BULK/evaluations/$RUN_ID"

mkdir -p "$RUN_DIR"

echo "$RUN_DIR" \
  > "$BULK/logs/n17_preppo_video_eval.latest"

echo "============================================================"
echo "GR00T N1.7 LIBERO-Spatial SFT baseline video evaluation"
echo "time:       $(date)"
echo "GPU:        physical 2,3"
echo "checkpoint: original SFT checkpoint"
echo "run dir:    $RUN_DIR"
echo "============================================================"

cd "$RLINF"

python "$EMBODIED_PATH/eval_embodied_agent_video.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$RUN_DIR"

echo
echo "N17_PREPPO_VIDEO_EVAL_COMPLETE"
