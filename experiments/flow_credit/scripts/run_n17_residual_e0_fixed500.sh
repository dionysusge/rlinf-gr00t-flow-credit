#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 2>/dev/null || true

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"
CONFIG_NAME=libero_spatial_n17_residual_fixed_eval_gpu01
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
export RAY_TMPDIR="$BULK/tmp/ray-residual-e0"
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RESIDUAL_DIAG_DIR 2>/dev/null || true

mkdir -p "$RAY_TMPDIR" "$BULK/evaluations" "$BULK/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT="$BULK/evaluations/n17_residual_e0_zero_fixed500_${STAMP}"
mkdir -p "$EVAL_ROOT"
echo "$EVAL_ROOT" > "$BULK/logs/n17_residual_e0.latest"

declare -a SET_NAMES=(setA setB setC setD setE)
declare -a OFFSETS=(0 10 20 30 40)

cd "$RLINF"
for index in "${!SET_NAMES[@]}"; do
    set_name=${SET_NAMES[$index]}
    offset=${OFFSETS[$index]}
    output_dir="$EVAL_ROOT/$set_name"
    mkdir -p "$output_dir"
    echo "[E0] evaluating $set_name (trial offset $offset)"
    set +e
    EVAL_LABEL="e0-zero-$set_name" python "$ENTRY" \
        --config-path "$EMBODIED_PATH/config" \
        --config-name "$CONFIG_NAME" \
        runner.resume_dir=null \
        "++env.eval.eval_reset_offset=$offset" \
        ++env.eval.eval_reset_limit=100 \
        runner.logger.log_path="$output_dir" \
        runner.logger.experiment_name="e0-zero-$set_name" \
        2>&1 | tee "$output_dir/evaluation.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" > "$output_dir/exit_code.txt"
    ray stop --force >/dev/null 2>&1 || true
    if [[ "$rc" -ne 0 ]]; then
        exit "$rc"
    fi
done

python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
    --input "setA=$EVAL_ROOT/setA" \
    --input "setB=$EVAL_ROOT/setB" \
    --input "setC=$EVAL_ROOT/setC" \
    --input "setD=$EVAL_ROOT/setD" \
    --input "setE=$EVAL_ROOT/setE" \
    --output-dir "$EVAL_ROOT/aggregate"

python - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "aggregate" / "summary.json").read_text())
expected_successes = 444
if summary["num_trials"] != 500:
    raise SystemExit(f"E0 failed: expected 500 trials, got {summary['num_trials']}")
if summary["successes"] != expected_successes:
    (root / "E0_FAIL").write_text(
        f"expected {expected_successes}/500, got {summary['successes']}/500\n"
    )
    raise SystemExit(
        "E0 failed zero-residual equivalence: "
        f"expected {expected_successes}/500, got {summary['successes']}/500"
    )
(root / "E0_PASS").write_text("zero residual reproduced 444/500\n")
print("E0_PASS: zero residual reproduced 444/500")
PY

echo "N17_RESIDUAL_E0_FIXED500_COMPLETE"
