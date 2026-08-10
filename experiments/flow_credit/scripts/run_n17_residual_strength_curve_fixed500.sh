#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?Usage: $0 /absolute/path/to/global_step_N}
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
export RAY_TMPDIR=/mnt/models/gzw/raytmp/strength
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RAY_ADDRESS 2>/dev/null || true

E0_ROOT=$(cat "$BULK/logs/n17_residual_e0.latest")
test -f "$E0_ROOT/aggregate/trials.csv"
STAMP=$(date +%Y%m%d_%H%M%S)
WANDB_EVIDENCE_RUN_ID="n17-residual-strength-$STAMP"
ROOT="$BULK/evaluations/n17_residual_strength_fixed500_${STAMP}"
mkdir -p "$ROOT" "$RAY_TMPDIR"
python "$RLINF/experiments/flow_credit/analysis/validate_ray_tmpdir.py" "$RAY_TMPDIR"
echo "$ROOT" > "$BULK/logs/n17_residual_strength.latest"
printf '%s\n' "$CHECKPOINT" > "$ROOT/checkpoint.txt"
printf '%s\n' "$E0_ROOT" > "$ROOT/e0_evaluation_root.txt"
git -C "$RLINF" rev-parse HEAD > "$ROOT/git_commit.txt"
git -C "$RLINF" status --short > "$ROOT/git_status.txt"

declare -a LAMBDAS=(0 0.25 0.5 1.0)
declare -a SET_NAMES=(setA setB setC setD setE)
declare -a OFFSETS=(0 10 20 30 40)
cd "$RLINF"

for lambda in "${LAMBDAS[@]}"; do
    lambda_name=${lambda/./p}
    lambda_root="$ROOT/lambda_$lambda_name"
    mkdir -p "$lambda_root"
    inputs=()
    heldout_inputs=()
    for index in "${!SET_NAMES[@]}"; do
        set_name=${SET_NAMES[$index]}
        offset=${OFFSETS[$index]}
        output_dir="$lambda_root/$set_name"
        mkdir -p "$output_dir"
        if [[ "${DUMP_RESIDUAL_DIAGNOSTICS:-1}" == "1" && "$lambda" == "1.0" ]]; then
            export RESIDUAL_DIAG_DIR="$output_dir/residual_diagnostics"
        else
            unset RESIDUAL_DIAG_DIR 2>/dev/null || true
        fi
        EVAL_LABEL="residual-lambda$lambda-$set_name" python "$ENTRY" \
            --config-path "$EMBODIED_PATH/config" \
            --config-name "$CONFIG_NAME" \
            runner.resume_dir="$CHECKPOINT" \
            actor.model.rl_head_config.residual_policy.force_zero=false \
            actor.model.rl_head_config.residual_policy.eval_scale="$lambda" \
            "++env.eval.eval_reset_offset=$offset" \
            ++env.eval.eval_reset_limit=100 \
            runner.logger.log_path="$output_dir" \
            runner.logger.experiment_name="residual-lambda$lambda-$set_name" \
            2>&1 | tee "$output_dir/evaluation.log"
        inputs+=(--input "$set_name=$output_dir")
        if [[ "$set_name" != "setA" ]]; then
            heldout_inputs+=(--input "$set_name=$output_dir")
        fi
    done
    python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
        "${inputs[@]}" --output-dir "$lambda_root/aggregate"
    python experiments/flow_credit/analysis/analyze_residual_pairing.py \
        --base "$E0_ROOT/aggregate/trials.csv" \
        --candidate "$lambda_root/aggregate/trials.csv" \
        --output-dir "$lambda_root/pairing"
    if [[ "$lambda" == "1.0" ]]; then
        for set_name in "${SET_NAMES[@]}"; do
            output_dir="$lambda_root/$set_name"
            if [[ -d "$output_dir/residual_diagnostics" ]]; then
                python experiments/flow_credit/analysis/analyze_residual_diagnostics.py \
                    --diagnostic-dir "$output_dir/residual_diagnostics" \
                    --trials-csv "$output_dir/trials.csv" \
                    --pairing-csv "$lambda_root/pairing/pairing.csv" \
                    --output-dir "$output_dir/residual_analysis"
            fi
        done
    fi
    python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
        "${heldout_inputs[@]}" --output-dir "$lambda_root/heldout_BtoE"
    python experiments/flow_credit/analysis/analyze_residual_pairing.py \
        --base "$E0_ROOT/aggregate_heldout_BtoE/trials.csv" \
        --candidate "$lambda_root/heldout_BtoE/trials.csv" \
        --output-dir "$lambda_root/heldout_BtoE_pairing"
done

python - "$ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for directory in root.glob("lambda_*"):
    value = float(directory.name.removeprefix("lambda_").replace("p", "."))
    metrics = json.loads((directory / "aggregate" / "summary.json").read_text())
    pairing = json.loads((directory / "pairing" / "summary.json").read_text())
    heldout_metrics = json.loads(
        (directory / "heldout_BtoE" / "summary.json").read_text()
    )
    heldout_pairing = json.loads(
        (directory / "heldout_BtoE_pairing" / "summary.json").read_text()
    )
    rows.append({
        "lambda": value,
        "success_rate": metrics["success_rate"],
        "reward": metrics["reward"],
        "episode_length": metrics["episode_length"],
        "rescue": pairing["rescue"],
        "harm": pairing["harm"],
        "net_rescue": pairing["net_rescue"],
        "mcnemar_exact_pvalue": pairing["mcnemar_exact_pvalue"],
        "heldout_success_rate": heldout_metrics["success_rate"],
        "heldout_reward": heldout_metrics["reward"],
        "heldout_episode_length": heldout_metrics["episode_length"],
        "heldout_rescue": heldout_pairing["rescue"],
        "heldout_harm": heldout_pairing["harm"],
        "heldout_net_rescue": heldout_pairing["net_rescue"],
        "heldout_mcnemar_exact_pvalue": heldout_pairing["mcnemar_exact_pvalue"],
    })
rows.sort(key=lambda row: row["lambda"])
zero_row = next(row for row in rows if row["lambda"] == 0.0)
if zero_row["rescue"] or zero_row["harm"]:
    (root / "LAMBDA0_FAIL").write_text(
        "lambda=0 did not reproduce the seeded E0 base: "
        f"rescue={zero_row['rescue']}, harm={zero_row['harm']}\n",
        encoding="utf-8",
    )
    raise SystemExit(
        "Strength-curve implementation check failed at lambda=0: "
        f"rescue={zero_row['rescue']}, harm={zero_row['harm']}"
    )
(root / "LAMBDA0_PASS").write_text(
    "lambda=0 exactly reproduced all seeded E0 outcomes\n", encoding="utf-8"
)
with (root / "strength_curve.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY

python experiments/flow_credit/analysis/log_evidence_to_wandb.py \
    --kind strength_curve \
    --root "$ROOT" \
    --name "Residual-Strength-Curve-Fixed500-$STAMP" \
    --run-id "$WANDB_EVIDENCE_RUN_ID"

echo "N17_RESIDUAL_STRENGTH_FIXED500_COMPLETE"
