#!/usr/bin/env bash
set -euo pipefail

OFFSET=${1:-0}
LABEL=${2:-step600}
NUM_ENVS=${3:-2}

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"

CONFIG_NAME=libero_spatial_n17_fixed100_eval_gpu23
ENTRY="$RLINF/examples/embodiment/eval_embodied_agent_fixed.py"

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

export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/tmp/r17e_apps
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"

RAY_REAL="$BULK/tmp/ray-fixed-eval"

mkdir -p \
  "$RAY_REAL" \
  "$BULK/evaluations" \
  "$BULK/logs"

if [[ ! -e "$RAY_TMPDIR" ]]; then
    ln -s "$RAY_REAL" "$RAY_TMPDIR"
fi

unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true

RUN_DIR=$(
    find "$BULK/runs" \
        -maxdepth 1 \
        -type d \
        -name 'n17_flow_sde_ppo_gpu23_fp32master_*' \
    | sort \
    | tail -1
)

CKPT_ROOT=$(
    find "$RUN_DIR" \
        -type d \
        -path '*/checkpoints' \
        -print \
    | head -1
)

case "$LABEL" in
    sft_base)
        RESUME_OVERRIDE="runner.resume_dir=null"
        ;;
    step400)
        RESUME_OVERRIDE="runner.resume_dir=$CKPT_ROOT/global_step_400"
        ;;
    step600)
        RESUME_OVERRIDE="runner.resume_dir=$CKPT_ROOT/global_step_600"
        ;;
    *)
        echo "Unknown LABEL=$LABEL"
        exit 1
        ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)

OUT="$BULK/evaluations/flow_geometry_probe_${LABEL}_offset${OFFSET}_${STAMP}"

mkdir -p "$OUT/raw"

echo "$OUT" > "$BULK/logs/flow_geometry_probe.latest"

export FLOW_DIAG_DIR="$OUT/raw"
export FLOW_DIAG_MAX_CALLS=128
export FLOW_DIAG_LABEL="$LABEL"
export FLOW_DIAG_RESET_OFFSET="$OFFSET"

cd "$RLINF"

echo "============================================================"
echo "Flow geometry probe"
echo "model:       $LABEL"
echo "offset:      $OFFSET"
echo "num envs:    $NUM_ENVS"
echo "raw output:  $FLOW_DIAG_DIR"
echo "============================================================"

set +e

python "$ENTRY" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    "$RESUME_OVERRIDE" \
    "env.eval.total_num_envs=$NUM_ENVS" \
    "++env.eval.eval_reset_offset=$OFFSET" \
    "++env.eval.eval_reset_limit=$NUM_ENVS" \
    "runner.logger.log_path=$OUT" \
    "runner.logger.experiment_name=flow-geometry-${LABEL}-offset${OFFSET}" \
    2>&1 | tee "$OUT/evaluation.log"

rc=${PIPESTATUS[0]}

set -e

echo "$rc" > "$OUT/exit_code.txt"

ray stop --force >/dev/null 2>&1 || true

echo
echo "OUTPUT_DIR=$OUT"
echo "RAW_DIR=$OUT/raw"
echo "EXIT_CODE=$rc"

exit "$rc"
