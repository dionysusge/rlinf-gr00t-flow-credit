#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?Usage: $0 /absolute/path/to/full_ppo/global_step_600}
PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"
CONFIG_NAME=libero_spatial_n17_residual_fixed_eval_gpu45
ENTRY="$RLINF/examples/embodiment/eval_embodied_agent_fixed.py"
test -d "$CHECKPOINT/actor"

source "$PROJECT/scripts/activate_rlinf.sh"
export EMBODIED_PATH="$RLINF/examples/embodiment"
export REPO_PATH="$RLINF"
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
export WANDB_ENTITY=liwuyu-cloudbutterfly
export WANDB_PROJECT=GR00T-Residual-RL
export WANDB_RUN_GROUP=Residual-Locality-Evidence
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RLINF_FORCE_LOCAL_RAY=1
# Keep this path short: Ray embeds a long session name below it and Linux
# AF_UNIX socket paths are limited to 107 bytes.
export RAY_TMPDIR=/mnt/models/gzw/raytmp/fullppo
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RAY_ADDRESS 2>/dev/null || true
unset RESIDUAL_DIAG_DIR 2>/dev/null || true

E0_ROOT=$(cat "$BULK/logs/n17_residual_e0.latest")
test -f "$E0_ROOT/aggregate/trials.csv"
test -f "$E0_ROOT/aggregate_heldout_BtoE/trials.csv"

STAMP=$(date +%Y%m%d_%H%M%S)
WANDB_EVIDENCE_RUN_ID="n17-fullppo-step600-paired-$STAMP"
ROOT="$BULK/evaluations/n17_fullppo_step600_fixed500_${STAMP}"
mkdir -p "$ROOT" "$RAY_TMPDIR"
python "$RLINF/experiments/flow_credit/analysis/validate_ray_tmpdir.py" "$RAY_TMPDIR"
echo "$ROOT" > "$BULK/logs/n17_fullppo_step600_fixed500.latest"
printf '%s\n' "$CHECKPOINT" > "$ROOT/checkpoint.txt"
printf '%s\n' "$E0_ROOT" > "$ROOT/e0_evaluation_root.txt"
git -C "$RLINF" rev-parse HEAD > "$ROOT/git_commit.txt"
git -C "$RLINF" status --short > "$ROOT/git_status.txt"

declare -a SET_NAMES=(setA setB setC setD setE)
declare -a OFFSETS=(0 10 20 30 40)
inputs=()
heldout_inputs=()
cd "$RLINF"
for index in "${!SET_NAMES[@]}"; do
    set_name=${SET_NAMES[$index]}
    offset=${OFFSETS[$index]}
    output_dir="$ROOT/$set_name"
    mkdir -p "$output_dir"
    echo "[Full PPO step600] evaluating $set_name (trial offset $offset)"
    EVAL_LABEL="fullppo-step600-$set_name" python "$ENTRY" \
        --config-path "$EMBODIED_PATH/config" \
        --config-name "$CONFIG_NAME" \
        runner.resume_dir="$CHECKPOINT" \
        actor.model.rl_head_config.residual_policy.enabled=false \
        "++env.eval.eval_reset_offset=$offset" \
        ++env.eval.eval_reset_limit=100 \
        runner.logger.log_path="$output_dir" \
        runner.logger.experiment_name="fullppo-step600-$set_name" \
        2>&1 | tee "$output_dir/evaluation.log"
    inputs+=(--input "$set_name=$output_dir")
    if [[ "$set_name" != "setA" ]]; then
        heldout_inputs+=(--input "$set_name=$output_dir")
    fi
done

python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
    "${inputs[@]}" --output-dir "$ROOT/aggregate"
python experiments/flow_credit/analysis/analyze_residual_pairing.py \
    --base "$E0_ROOT/aggregate/trials.csv" \
    --candidate "$ROOT/aggregate/trials.csv" \
    --output-dir "$ROOT/pairing"
python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
    "${heldout_inputs[@]}" --output-dir "$ROOT/aggregate_heldout_BtoE"
python experiments/flow_credit/analysis/analyze_residual_pairing.py \
    --base "$E0_ROOT/aggregate_heldout_BtoE/trials.csv" \
    --candidate "$ROOT/aggregate_heldout_BtoE/trials.csv" \
    --output-dir "$ROOT/heldout_BtoE_pairing"

python experiments/flow_credit/analysis/log_evidence_to_wandb.py \
    --kind fullppo \
    --root "$ROOT" \
    --name "FullPPO-Step600-Paired-Fixed500-$STAMP" \
    --run-id "$WANDB_EVIDENCE_RUN_ID"

echo "N17_FULLPPO_STEP600_FIXED500_COMPLETE"
