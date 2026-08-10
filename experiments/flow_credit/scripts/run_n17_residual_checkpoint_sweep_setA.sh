#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 2>/dev/null || true
PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"
CONFIG_NAME=libero_spatial_n17_residual_fixed_eval_gpu45
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
export WANDB_MODE=online
export WANDB_ENTITY=liwuyu-cloudbutterfly
export WANDB_PROJECT=GR00T-Residual-RL
export WANDB_RUN_GROUP=Residual-Locality-Evidence
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RLINF_FORCE_LOCAL_RAY=1
# Keep this path short: Ray embeds a long session name below it and Linux
# AF_UNIX socket paths are limited to 107 bytes.
export RAY_TMPDIR=/mnt/models/gzw/raytmp/sweep
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RAY_ADDRESS 2>/dev/null || true

RUN_DIR=$(cat "$BULK/logs/n17_residual_ppo.latest")
E0_ROOT=$(cat "$BULK/logs/n17_residual_e0.latest")
CKPT_ROOT=$(find "$RUN_DIR" -type d -path '*/checkpoints' -print | head -1)
test -d "$CKPT_ROOT"
test -f "$E0_ROOT/setA/trials.csv"

STAMP=$(date +%Y%m%d_%H%M%S)
WANDB_EVIDENCE_RUN_ID="n17-residual-checkpoint-sweep-$STAMP"
SWEEP_ROOT="$BULK/evaluations/n17_residual_setA_checkpoint_sweep_${STAMP}"
mkdir -p "$SWEEP_ROOT" "$RAY_TMPDIR"
python "$RLINF/experiments/flow_credit/analysis/validate_ray_tmpdir.py" "$RAY_TMPDIR"
echo "$SWEEP_ROOT" > "$BULK/logs/n17_residual_setA_sweep.latest"
printf '%s\n' "$RUN_DIR" > "$SWEEP_ROOT/training_run_dir.txt"
printf '%s\n' "$CKPT_ROOT" > "$SWEEP_ROOT/checkpoint_root.txt"
printf '%s\n' "$E0_ROOT" > "$SWEEP_ROOT/e0_evaluation_root.txt"
git -C "$RLINF" rev-parse HEAD > "$SWEEP_ROOT/git_commit.txt"
git -C "$RLINF" status --short > "$SWEEP_ROOT/git_status.txt"
declare -a STEPS=(30 60 90 120 150)

cd "$RLINF"
for step in "${STEPS[@]}"; do
    checkpoint="$CKPT_ROOT/global_step_$step"
    test -d "$checkpoint/actor"
    output_dir="$SWEEP_ROOT/step$step"
    mkdir -p "$output_dir"
    if [[ "${DUMP_RESIDUAL_DIAGNOSTICS:-1}" == "1" ]]; then
        export RESIDUAL_DIAG_DIR="$output_dir/residual_diagnostics"
    else
        unset RESIDUAL_DIAG_DIR 2>/dev/null || true
    fi
    EVAL_LABEL="residual-step$step-setA" python "$ENTRY" \
        --config-path "$EMBODIED_PATH/config" \
        --config-name "$CONFIG_NAME" \
        runner.resume_dir="$checkpoint" \
        actor.model.rl_head_config.residual_policy.force_zero=false \
        actor.model.rl_head_config.residual_policy.eval_scale=1.0 \
        ++env.eval.eval_reset_offset=0 \
        ++env.eval.eval_reset_limit=100 \
        runner.logger.log_path="$output_dir" \
        runner.logger.experiment_name="residual-step$step-setA" \
        2>&1 | tee "$output_dir/evaluation.log"
    python experiments/flow_credit/analysis/analyze_residual_pairing.py \
        --base "$E0_ROOT/setA/trials.csv" \
        --candidate "$output_dir/trials.csv" \
        --output-dir "$output_dir/pairing"
    if [[ -d "$output_dir/residual_diagnostics" ]]; then
        python experiments/flow_credit/analysis/analyze_residual_diagnostics.py \
            --diagnostic-dir "$output_dir/residual_diagnostics" \
            --trials-csv "$output_dir/trials.csv" \
            --pairing-csv "$output_dir/pairing/pairing.csv" \
            --output-dir "$output_dir/residual_analysis"
    fi
done

python - "$SWEEP_ROOT" "$CKPT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint_root = Path(sys.argv[2])
rows = []
for directory in sorted(root.glob("step*"), key=lambda path: int(path.name[4:])):
    metrics = json.loads((directory / "metrics.json").read_text())["metrics"]
    pairing = json.loads((directory / "pairing" / "summary.json").read_text())
    rows.append({
        "step": int(directory.name[4:]),
        "environment_transitions": int(directory.name[4:]) * 4096,
        "success_rate": float(metrics["success_once"]),
        "reward": float(metrics["reward"]),
        "episode_length": float(metrics["episode_len"]),
        "rescue": pairing["rescue"],
        "harm": pairing["harm"],
        "net_rescue": pairing["net_rescue"],
        "mcnemar_exact_pvalue": pairing["mcnemar_exact_pvalue"],
    })
with (root / "checkpoint_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
ranked = sorted(
    rows,
    key=lambda row: (
        -row["success_rate"],
        row["harm"],
        row["episode_length"],
        row["step"],
    ),
)
for rank, row in enumerate(ranked, start=1):
    row["selection_rank"] = rank
with (root / "checkpoint_selection_ranking.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
    writer.writeheader()
    writer.writerows(ranked)
best = ranked[0]
best_checkpoint = checkpoint_root / f"global_step_{best['step']}"
selection = {
    "criterion": [
        "maximize Set-A success_rate",
        "minimize harm",
        "minimize episode_length",
        "prefer earlier step",
    ],
    "best": best,
    "best_checkpoint": str(best_checkpoint),
}
(root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
(root / "best_checkpoint.txt").write_text(str(best_checkpoint) + "\n")
print(json.dumps({"rows": rows, "selection": selection}, indent=2))
PY

python experiments/flow_credit/analysis/log_evidence_to_wandb.py \
    --kind checkpoint_sweep \
    --root "$SWEEP_ROOT" \
    --name "Residual-Checkpoint-Sweep-SetA-$STAMP" \
    --run-id "$WANDB_EVIDENCE_RUN_ID"

echo "N17_RESIDUAL_CHECKPOINT_SWEEP_SETA_COMPLETE"
