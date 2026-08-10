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
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
export WANDB_ENTITY=liwuyu-cloudbutterfly
export WANDB_PROJECT=GR00T-Residual-RL
export WANDB_RUN_GROUP=Residual-Locality-Evidence
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RLINF_FORCE_LOCAL_RAY=1
# Keep this path short: Ray embeds a long session name below it and Linux
# AF_UNIX socket paths are limited to 107 bytes.
export RAY_TMPDIR=/mnt/models/gzw/raytmp/e0
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RAY_ADDRESS 2>/dev/null || true
unset RESIDUAL_DIAG_DIR 2>/dev/null || true

mkdir -p "$RAY_TMPDIR" "$BULK/evaluations" "$BULK/logs"
python "$RLINF/experiments/flow_credit/analysis/validate_ray_tmpdir.py" "$RAY_TMPDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
WANDB_EVIDENCE_RUN_ID="n17-residual-e0-$STAMP"
EVAL_ROOT="$BULK/evaluations/n17_residual_e0_seeded_fixed500_${STAMP}"
mkdir -p "$EVAL_ROOT"
echo "$EVAL_ROOT" > "$BULK/logs/n17_residual_e0.latest"
git -C "$RLINF" rev-parse HEAD > "$EVAL_ROOT/git_commit.txt"
git -C "$RLINF" status --short > "$EVAL_ROOT/git_status.txt"
python - "$EVAL_ROOT/e0_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "gpus": [4, 5],
            "env_seed": 0,
            "rollout_seed": 1234,
            "rollout_rank_seed_rule": "rollout_seed + rollout_rank",
            "sets": ["setA", "setB", "setC", "setD", "setE"],
            "trials_per_variant": 500,
            "variants": {
                "base": {"residual_enabled": False, "force_zero": False},
                "zero": {"residual_enabled": True, "force_zero": True},
            },
            "historical_unseeded_reference": "444/500",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

declare -a SET_NAMES=(setA setB setC setD setE)
declare -a OFFSETS=(0 10 20 30 40)

cd "$RLINF"
EVIDENCE_UPLOADED=0
upload_evidence() {
    python experiments/flow_credit/analysis/log_evidence_to_wandb.py \
        --kind e0 \
        --root "$EVAL_ROOT" \
        --name "Residual-E0-Seeded-Equivalence-$STAMP" \
        --run-id "$WANDB_EVIDENCE_RUN_ID"
}
on_exit() {
    rc=$?
    trap - EXIT
    if [[ "$EVIDENCE_UPLOADED" -eq 0 ]]; then
        echo "[E0] uploading available evidence to W&B before exit"
        set +e
        upload_evidence
        set -e
    fi
    exit "$rc"
}
trap on_exit EXIT

run_variant() {
    variant=$1
    variant_root=$2
    residual_enabled=$3
    force_zero=$4
    for index in "${!SET_NAMES[@]}"; do
        set_name=${SET_NAMES[$index]}
        offset=${OFFSETS[$index]}
        output_dir="$variant_root/$set_name"
        mkdir -p "$output_dir"
        echo "[E0] evaluating $variant/$set_name (trial offset $offset)"
        set +e
        EVAL_LABEL="e0-$variant-$set_name" python "$ENTRY" \
            --config-path "$EMBODIED_PATH/config" \
            --config-name "$CONFIG_NAME" \
            runner.resume_dir=null \
            "++env.eval.eval_reset_offset=$offset" \
            ++env.eval.eval_reset_limit=100 \
            "++actor.model.rl_head_config.residual_policy.enabled=$residual_enabled" \
            "++actor.model.rl_head_config.residual_policy.force_zero=$force_zero" \
            "++rollout.model.rl_head_config.residual_policy.enabled=$residual_enabled" \
            "++rollout.model.rl_head_config.residual_policy.force_zero=$force_zero" \
            runner.logger.log_path="$output_dir" \
            runner.logger.experiment_name="e0-$variant-$set_name" \
            2>&1 | tee "$output_dir/evaluation.log"
        rc=${PIPESTATUS[0]}
        set -e
        echo "$rc" > "$output_dir/exit_code.txt"
        if [[ "$rc" -ne 0 ]]; then
            exit "$rc"
        fi
    done
}

aggregate_variant() {
    variant_root=$1
    python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
        --input "setA=$variant_root/setA" \
        --input "setB=$variant_root/setB" \
        --input "setC=$variant_root/setC" \
        --input "setD=$variant_root/setD" \
        --input "setE=$variant_root/setE" \
        --output-dir "$variant_root/aggregate"
    python experiments/flow_credit/analysis/aggregate_fixed_trial_evaluations.py \
        --input "setB=$variant_root/setB" \
        --input "setC=$variant_root/setC" \
        --input "setD=$variant_root/setD" \
        --input "setE=$variant_root/setE" \
        --output-dir "$variant_root/aggregate_heldout_BtoE"
}

# The scientific invariant is same-pipeline equivalence under the same model
# RNG stream. A historical unseeded 444/500 result is only a reference, not a
# valid exact gate for a newly seeded flow-policy evaluation.
run_variant base "$EVAL_ROOT/base" false false
run_variant zero "$EVAL_ROOT" true true
aggregate_variant "$EVAL_ROOT/base"
aggregate_variant "$EVAL_ROOT"

python experiments/flow_credit/analysis/analyze_residual_pairing.py \
    --base "$EVAL_ROOT/base/aggregate/trials.csv" \
    --candidate "$EVAL_ROOT/aggregate/trials.csv" \
    --output-dir "$EVAL_ROOT/base_zero_pairing"

python experiments/flow_credit/analysis/analyze_residual_pairing.py \
    --base "$EVAL_ROOT/base/aggregate_heldout_BtoE/trials.csv" \
    --candidate "$EVAL_ROOT/aggregate_heldout_BtoE/trials.csv" \
    --output-dir "$EVAL_ROOT/base_zero_pairing_heldout_BtoE"

python - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "base" / "aggregate" / "summary.json").read_text())
zero = json.loads((root / "aggregate" / "summary.json").read_text())
pairing = json.loads((root / "base_zero_pairing" / "summary.json").read_text())
reference = {
    "historical_unseeded_successes": 444,
    "historical_unseeded_num_trials": 500,
    "seeded_base_successes": base["successes"],
    "seeded_zero_successes": zero["successes"],
}
(root / "historical_reference.json").write_text(
    json.dumps(reference, indent=2) + "\n", encoding="utf-8"
)
if base["num_trials"] != 500 or zero["num_trials"] != 500:
    raise SystemExit(
        f"E0 failed: expected 500+500 trials, got "
        f"{base['num_trials']}+{zero['num_trials']}"
    )
if pairing["rescue"] or pairing["harm"]:
    (root / "E0_FAIL").write_text(
        "seeded residual-disabled and force-zero outcomes differ: "
        f"rescue={pairing['rescue']}, harm={pairing['harm']}\n",
        encoding="utf-8",
    )
    raise SystemExit(
        "E0 failed paired zero-residual equivalence: "
        f"rescue={pairing['rescue']}, harm={pairing['harm']}"
    )
print(
    "E0 base/zero paired equivalence passed: "
    f"{zero['successes']}/500 (historical unseeded reference: 444/500)"
)
PY

# E0-R: repeat the same Set-A fixed resets in an independent process. This is
# a hard gate. With rollout.seed configured, any mismatch means the inference
# pipeline is not repeatable enough for per-trial rescue/harm analysis.
REPEAT_DIR="$EVAL_ROOT/setA_repeat"
mkdir -p "$REPEAT_DIR"
echo "[E0-R] repeating Set-A (trial offset 0) for per-trial repeatability"
set +e
EVAL_LABEL="e0-zero-setA-repeat" python "$ENTRY" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.resume_dir=null \
    ++env.eval.eval_reset_offset=0 \
    ++env.eval.eval_reset_limit=100 \
    ++actor.model.rl_head_config.residual_policy.enabled=true \
    ++actor.model.rl_head_config.residual_policy.force_zero=true \
    ++rollout.model.rl_head_config.residual_policy.enabled=true \
    ++rollout.model.rl_head_config.residual_policy.force_zero=true \
    runner.logger.log_path="$REPEAT_DIR" \
    runner.logger.experiment_name="e0-zero-setA-repeat" \
    2>&1 | tee "$REPEAT_DIR/evaluation.log"
rc=${PIPESTATUS[0]}
set -e
echo "$rc" > "$REPEAT_DIR/exit_code.txt"
if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
fi

python experiments/flow_credit/analysis/analyze_residual_pairing.py \
    --base "$EVAL_ROOT/setA/trials.csv" \
    --candidate "$REPEAT_DIR/trials.csv" \
    --output-dir "$EVAL_ROOT/setA_repeatability"

python - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads(
    (root / "setA_repeatability" / "summary.json").read_text(encoding="utf-8")
)
repeatable = summary["rescue"] == 0 and summary["harm"] == 0
payload = {"repeatable": repeatable, **summary}
(root / "E0_REPEATABILITY.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
marker = root / ("E0_REPEATABLE" if repeatable else "E0_REPEATABILITY_WARNING")
marker.write_text(
    "Set-A outcomes are exactly repeatable\n"
    if repeatable
    else "Set-A outcomes changed across zero-residual repeats; inspect pairing\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
if not repeatable:
    (root / "E0_FAIL").write_text(
        "seeded Set-A repeatability failed\n", encoding="utf-8"
    )
    raise SystemExit("E0 failed: seeded Set-A outcomes were not exactly repeatable")
(root / "E0_PASS").write_text(
    "seeded base/zero fixed500 equivalence and Set-A repeatability passed\n",
    encoding="utf-8",
)
print("E0_PASS: seeded base/zero equivalence and repeatability passed")
PY

upload_evidence
EVIDENCE_UPLOADED=1

echo "N17_RESIDUAL_E0_FIXED500_COMPLETE"
