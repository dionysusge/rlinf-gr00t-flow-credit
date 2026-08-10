#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"
CONFIG_NAME=libero_spatial_residual_ppo_gr00t_n1d7_h200_gpu01

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
export WANDB_PROJECT=GR00T-Residual-RL
export WANDB_RUN_GROUP=Residual-PPO-0.1-GR00T-N1.7-LIBERO-Spatial
export WANDB_DIR="$BULK/wandb"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="$BULK/tmp/ray-residual-ppo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true

E0_ROOT=$(cat "$BULK/logs/n17_residual_e0.latest" 2>/dev/null || true)
if [[ "${ALLOW_UNVERIFIED_E0:-0}" != "1" ]]; then
    test -n "$E0_ROOT"
    test -f "$E0_ROOT/E0_PASS"
fi

mkdir -p "$RAY_TMPDIR" "$BULK/runs" "$BULK/logs" "$BULK/wandb"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="n17_residual_ppo_a01_gpu01_${STAMP}"
RUN_NAME="Residual-PPO-0.1-GR00T-N1.7-LIBERO-Spatial-H200x2-${STAMP}"
RUN_DIR="$BULK/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$BULK/logs/n17_residual_ppo.latest"

cp "$RLINF/examples/embodiment/config/${CONFIG_NAME}.yaml" "$RUN_DIR/launch_config.yaml"
git -C "$RLINF" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$RLINF" status --short > "$RUN_DIR/git_status.txt"
printf '%s\n' "$E0_ROOT" > "$RUN_DIR/e0_evaluation_root.txt"

echo "============================================================"
echo "Residual PPO E1"
echo "run:                 $RUN_NAME"
echo "GPUs:                0,1"
echo "base GR00T:          frozen"
echo "residual bound:      0.1"
echo "actor/critic LR:     1e-4 / 1e-4"
echo "transitions/update:  4096"
echo "checkpoint steps:    30,60,90,120,150"
echo "checkpoint approx:   123K,246K,369K,492K,614K transitions"
echo "E0 root:             $E0_ROOT"
echo "run dir:             $RUN_DIR"
echo "============================================================"

cd "$RLINF"
python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$RUN_DIR" \
    runner.logger.experiment_name="$RUN_NAME" \
    2>&1 | tee "$RUN_DIR/training.log"

echo "N17_RESIDUAL_PPO_E1_COMPLETE"
