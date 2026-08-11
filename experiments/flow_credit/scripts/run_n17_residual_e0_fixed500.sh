#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 2>/dev/null || true

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"
CONFIG_NAME=libero_spatial_n17_residual_fixed_eval_gpu01
ENTRY="$RLINF/examples/embodiment/eval_embodied_agent_fixed.py"

# Keep the lock in a small outer process. --close prevents the Python driver
# and Ray workers from inheriting its file descriptor.
if [[ "${E0_LOCK_HELD:-0}" != "1" ]]; then
    mkdir -p "$BULK/locks"
    SCRIPT_PATH=$(readlink -f "$0")
    set +e
    flock --close --nonblock --conflict-exit-code 73 \
        "$BULK/locks/n17_residual_e0.lock" \
        env E0_LOCK_HELD=1 bash "$SCRIPT_PATH" "$@"
    rc=$?
    set -e
    if [[ "$rc" -eq 73 ]]; then
        echo "Another E0 launcher is already using GPUs 0/1" >&2
    fi
    exit "$rc"
fi

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
# A dead rollout rank must not leave its peers blocked in Gloo for the default
# 180 minutes. Normal fixed100 evaluation takes only a few minutes.
export RLINF_TIMEOUT="${E0_COLLECTIVE_TIMEOUT_MINUTES:-15}"
# Keep this path short: Ray embeds a long session name below it and Linux
# AF_UNIX socket paths are limited to 107 bytes.
export RAY_TMPDIR=/mnt/models/gzw/raytmp/e0
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RAY_ADDRESS 2>/dev/null || true
unset RESIDUAL_DIAG_DIR 2>/dev/null || true

mkdir -p "$RAY_TMPDIR" "$BULK/evaluations" "$BULK/logs" "$BULK/locks"
python "$RLINF/experiments/flow_credit/analysis/validate_ray_tmpdir.py" "$RAY_TMPDIR"

export E0_MAX_ATTEMPTS="${E0_MAX_ATTEMPTS:-2}"
export E0_SET_TIMEOUT="${E0_SET_TIMEOUT:-45m}"
if [[ ! "$E0_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "E0_MAX_ATTEMPTS must be a positive integer" >&2
    exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT="${E0_RESUME_ROOT:-$BULK/evaluations/n17_residual_e0_seeded_fixed500_${STAMP}}"
mkdir -p "$EVAL_ROOT"
mapfile -t STALE_EVAL_PIDS < <(
    pgrep -f "[e]val_embodied_agent_fixed.py.*runner.logger.log_path=$EVAL_ROOT" \
        || true
)
if (( ${#STALE_EVAL_PIDS[@]} > 0 )); then
    echo "E0 cannot start: stale evaluator processes still target $EVAL_ROOT" >&2
    for stale_pid in "${STALE_EVAL_PIDS[@]}"; do
        ps -p "$stale_pid" -o pid,ppid,user,stat,lstart,cmd --no-headers >&2 \
            || true
    done
    echo "Terminate only the listed PIDs, then resume the same E0 root." >&2
    exit 74
fi
if [[ -f "$EVAL_ROOT/wandb_evidence_run_id.txt" ]]; then
    WANDB_EVIDENCE_RUN_ID=$(<"$EVAL_ROOT/wandb_evidence_run_id.txt")
else
    WANDB_EVIDENCE_RUN_ID="n17-residual-e0-$STAMP"
    printf '%s\n' "$WANDB_EVIDENCE_RUN_ID" \
        > "$EVAL_ROOT/wandb_evidence_run_id.txt"
fi
echo "$EVAL_ROOT" > "$BULK/logs/n17_residual_e0.latest"
git -C "$RLINF" rev-parse HEAD > "$EVAL_ROOT/git_commit.txt"
git -C "$RLINF" status --short > "$EVAL_ROOT/git_status.txt"
if [[ -f "$EVAL_ROOT/e0_manifest.json" \
    && ! -f "$EVAL_ROOT/e0_manifest.before_gpu01_resume.json" ]]; then
    cp "$EVAL_ROOT/e0_manifest.json" \
        "$EVAL_ROOT/e0_manifest.before_gpu01_resume.json"
fi
python - "$EVAL_ROOT/e0_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "gpus": [0, 1],
            "runtime": "ray_fsdp_parallel",
            "parallel_actor_workers": 2,
            "parallel_rollout_workers": 2,
            "env_seed": 0,
            "rollout_seed": 1234,
            "rollout_rank_seed_rule": "rollout_seed + rollout_rank",
            "collective_timeout_minutes": int(
                __import__("os").environ.get("RLINF_TIMEOUT", "15")
            ),
            "max_attempts_per_set": int(
                __import__("os").environ.get("E0_MAX_ATTEMPTS", "2")
            ),
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

python - "$EVAL_ROOT/e0_launch_history.jsonl" "$EVAL_ROOT" "$RLINF" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

history_path = Path(sys.argv[1])
root = Path(sys.argv[2])
repo = Path(sys.argv[3])
completed_before_launch = []
for exit_code_path in sorted(root.rglob("exit_code.txt")):
    try:
        if exit_code_path.read_text(encoding="utf-8").strip() == "0":
            completed_before_launch.append(str(exit_code_path.parent.relative_to(root)))
    except OSError:
        continue

try:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError) as error:
    git_commit = f"unavailable: {error}"

entry = {
    "started_at": datetime.now().astimezone().isoformat(),
    "gpus": [0, 1],
    "resume_root": os.environ.get("E0_RESUME_ROOT"),
    "git_commit": git_commit,
    "completed_sets_before_launch": completed_before_launch,
}
with history_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY

declare -a SET_NAMES=(setA setB setC setD setE)
declare -a OFFSETS=(0 100 200 300 400)

cd "$RLINF"
EVIDENCE_UPLOADED=0
upload_evidence() {
    python experiments/flow_credit/analysis/log_evidence_to_wandb.py \
        --kind e0 \
        --root "$EVAL_ROOT" \
        --name "Residual-E0-Seeded-Equivalence-$STAMP" \
        --run-id "$WANDB_EVIDENCE_RUN_ID"
}
upload_progress() {
    local upload_rc
    set +e
    upload_evidence
    upload_rc=$?
    set -e
    if [[ "$upload_rc" -ne 0 ]]; then
        echo "[E0] non-fatal W&B progress upload failure (rc=$upload_rc)" >&2
    fi
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

# Make the W&B run visible at launch. Completed set metrics append to the same
# stable run ID below.
upload_progress

validate_fixed_set() {
    local output_dir=$1
    local expected_offset=$2
    local expected_label=$3
    python - "$output_dir" "$expected_offset" "$expected_label" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_offset = int(sys.argv[2])
expected_label = sys.argv[3]
try:
    with (root / "trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    exit_code = (root / "exit_code.txt").read_text(encoding="utf-8").strip()
    ok = (
        exit_code == "0"
        and len(rows) == 100
        and len({(row["task_id"], row["trial_id"]) for row in rows}) == 100
        and {row["model"] for row in rows} == {expected_label}
        and payload["label"] == expected_label
        and payload["expected_trajectories"] == 100
        and payload["num_trial_records"] == 100
        and payload["num_unique_trials"] == 100
        and payload["eval_reset_offset"] == expected_offset
        and payload["eval_reset_limit"] == 100
    )
except (OSError, KeyError, ValueError, json.JSONDecodeError):
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

run_fixed_set() {
    local output_dir=$1
    local offset=$2
    local label=$3
    local residual_enabled=$4
    local force_zero=$5
    local attempt rc

    mkdir -p "$output_dir"
    if validate_fixed_set "$output_dir" "$offset" "$label"; then
        echo "[E0] reusing completed $label"
        upload_progress
        return 0
    fi

    for ((attempt = 1; attempt <= E0_MAX_ATTEMPTS; attempt++)); do
        echo "[E0] evaluating $label (offset=$offset, attempt=$attempt/$E0_MAX_ATTEMPTS)"
        set +e
        EVAL_LABEL="$label" timeout --signal=TERM --kill-after=120s \
            "$E0_SET_TIMEOUT" \
            python "$ENTRY" \
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
            runner.logger.experiment_name="$label" \
            2>&1 | tee "$output_dir/evaluation.log"
        rc=${PIPESTATUS[0]}
        set -e
        echo "$rc" > "$output_dir/exit_code.txt"
        echo "$rc" > "$output_dir/exit_code.attempt${attempt}.txt"

        if [[ "$rc" -eq 0 ]]; then
            if validate_fixed_set "$output_dir" "$offset" "$label"; then
                echo "[E0] validated $label: 100 unique fixed trials"
                upload_progress
                return 0
            fi
            rc=3
            echo "$rc" > "$output_dir/exit_code.txt"
        fi
        if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
            printf 'attempt=%s timeout_rc=%s\n' "$attempt" "$rc" \
                > "$output_dir/EVAL_TIMEOUT.attempt${attempt}"
        fi
        if ((attempt < E0_MAX_ATTEMPTS)); then
            echo "[E0] $label failed validation (rc=$rc); retrying with a fresh local Ray runtime"
            sleep 10
        fi
    done

    echo "[E0] $label failed after $E0_MAX_ATTEMPTS attempts" >&2
    return "${rc:-1}"
}

run_variant() {
    variant=$1
    variant_root=$2
    residual_enabled=$3
    force_zero=$4
    for index in "${!SET_NAMES[@]}"; do
        set_name=${SET_NAMES[$index]}
        offset=${OFFSETS[$index]}
        run_fixed_set \
            "$variant_root/$set_name" \
            "$offset" \
            "e0-$variant-$set_name" \
            "$residual_enabled" \
            "$force_zero"
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
echo "[E0-R] repeating Set-A (trial offset 0) for per-trial repeatability"
run_fixed_set "$REPEAT_DIR" 0 "e0-zero-setA-repeat" true true

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
