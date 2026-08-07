#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 2>/dev/null || true

OFFSET=${1:?Usage: $0 OFFSET SET_NAME}
SET_NAME=${2:?Usage: $0 OFFSET SET_NAME}

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

test -d "$CKPT_ROOT/global_step_600/actor"
test -d "$CKPT_ROOT/global_step_650/actor"

STAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT="$BULK/evaluations/n17_fixed100_${SET_NAME}_${STAMP}"

mkdir -p "$EVAL_ROOT"

echo "$EVAL_ROOT" \
    > "$BULK/logs/n17_fixed100_${SET_NAME}.latest"

echo "$CKPT_ROOT" > "$EVAL_ROOT/checkpoint_root.txt"
echo "$OFFSET" > "$EVAL_ROOT/eval_reset_offset.txt"

declare -a LABELS=(
    "sft_base"
    "step600"
    "step650"
)

declare -a RESUME_DIRS=(
    "null"
    "$CKPT_ROOT/global_step_600"
    "$CKPT_ROOT/global_step_650"
)

cd "$RLINF"

for index in "${!LABELS[@]}"
do
    label="${LABELS[$index]}"
    resume_dir="${RESUME_DIRS[$index]}"
    output_dir="$EVAL_ROOT/$label"
    log_file="$output_dir/evaluation.log"

    mkdir -p "$output_dir"

    if [[ "$resume_dir" == "null" ]]; then
        resume_override="runner.resume_dir=null"
    else
        resume_override="runner.resume_dir=$resume_dir"
    fi

    echo
    echo "============================================================"
    echo "evaluation set:  $SET_NAME"
    echo "reset offset:    $OFFSET"
    echo "model:           $label"
    echo "resume_dir:      $resume_dir"
    echo "output_dir:      $output_dir"
    echo "============================================================"

    set +e

    EVAL_LABEL="${SET_NAME}-${label}" \
    python "$ENTRY" \
        --config-path "$EMBODIED_PATH/config" \
        --config-name "$CONFIG_NAME" \
        "$resume_override" \
        "++env.eval.eval_reset_offset=$OFFSET" \
        "++env.eval.eval_reset_limit=100" \
        "runner.logger.log_path=$output_dir" \
        "runner.logger.experiment_name=${SET_NAME}-${label}" \
        2>&1 | tee "$log_file"

    rc=${PIPESTATUS[0]}

    set -e

    echo "$rc" > "$output_dir/exit_code.txt"

    ray stop --force >/dev/null 2>&1 || true
    sleep 3

    if [[ "$rc" -ne 0 ]]; then
        echo "[FAIL] evaluation failed: $label"
        exit "$rc"
    fi

    grep -q 'FIXED100_EVAL_COMPLETE' "$log_file" || {
        echo "[FAIL] completion marker missing: $label"
        exit 1
    }

    echo "[PASS] $SET_NAME $label"
done

python - "$EVAL_ROOT" "$SET_NAME" "$OFFSET" <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import re
import sys

root = Path(sys.argv[1])
set_name = sys.argv[2]
offset = int(sys.argv[3])

labels = [
    "sft_base",
    "step600",
    "step650",
]

pattern = re.compile(
    r"\[libero eval\] "
    r"task_id=(\d+), "
    r"trial_id=(\d+), "
    r"success=(True|False)"
)

metrics_rows = []
trial_rows = []
outcomes = {}

for label in labels:
    metrics_path = root / label / "metrics.json"
    log_path = root / label / "evaluation.log"

    payload = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    metrics = payload["metrics"]

    if int(metrics["num_trajectories"]) != 100:
        raise SystemExit(
            f"[FAIL] {label}: expected 100 trajectories"
        )

    metrics_rows.append(
        {
            "set": set_name,
            "offset": offset,
            "label": label,
            "success_once": float(metrics["success_once"]),
            "return": float(metrics["return"]),
            "reward": float(metrics["reward"]),
            "episode_len": float(metrics["episode_len"]),
            "num_trajectories": int(
                metrics["num_trajectories"]
            ),
        }
    )

    parsed = {}

    for line in log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        match = pattern.search(line)

        if not match:
            continue

        task_id = int(match.group(1))
        trial_id = int(match.group(2))
        success = match.group(3) == "True"

        parsed[(task_id, trial_id)] = success

    if len(parsed) != 100:
        raise SystemExit(
            f"[FAIL] {label}: expected 100 unique trial records, "
            f"got {len(parsed)}"
        )

    outcomes[label] = parsed

    for (task_id, trial_id), success in sorted(parsed.items()):
        trial_rows.append(
            {
                "set": set_name,
                "offset": offset,
                "label": label,
                "task_id": task_id,
                "trial_id": trial_id,
                "success": int(success),
            }
        )

with (root / "summary.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(metrics_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(metrics_rows)

with (root / "trials.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(trial_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(trial_rows)

task_rows = []

for label in labels:
    per_task = defaultdict(list)

    for (task_id, _), success in outcomes[label].items():
        per_task[task_id].append(success)

    for task_id, values in sorted(per_task.items()):
        task_rows.append(
            {
                "set": set_name,
                "label": label,
                "task_id": task_id,
                "successes": sum(values),
                "total": len(values),
                "success_rate": sum(values) / len(values),
            }
        )

with (root / "task_summary.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(task_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(task_rows)


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed

    if discordant == 0:
        return 1.0

    tail = min(improved, regressed)

    probability = sum(
        math.comb(discordant, value)
        for value in range(tail + 1)
    ) / (2 ** discordant)

    return min(1.0, 2.0 * probability)


comparison_rows = []
sft = outcomes["sft_base"]

for label in ["step600", "step650"]:
    model = outcomes[label]

    assert sft.keys() == model.keys()

    improved = sum(
        (not sft[key]) and model[key]
        for key in sft
    )

    regressed = sum(
        sft[key] and (not model[key])
        for key in sft
    )

    both_success = sum(
        sft[key] and model[key]
        for key in sft
    )

    both_fail = sum(
        (not sft[key]) and (not model[key])
        for key in sft
    )

    comparison_rows.append(
        {
            "set": set_name,
            "model": label,
            "both_success": both_success,
            "improved_over_sft": improved,
            "regressed_from_sft": regressed,
            "both_fail": both_fail,
            "paired_delta": (improved - regressed) / 100,
            "mcnemar_exact_p": exact_mcnemar_p(
                improved,
                regressed,
            ),
        }
    )

with (root / "paired_comparison.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(comparison_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(comparison_rows)

print()
print("===== evaluation summary =====")

for row in metrics_rows:
    print(
        f"{row['label']:10s} "
        f"success={row['success_once']:.4f} "
        f"reward={row['reward']:.6f} "
        f"ep_len={row['episode_len']:.2f}"
    )

print()
print("===== paired comparison against SFT =====")

for row in comparison_rows:
    print(
        f"{row['model']:10s} "
        f"improved={row['improved_over_sft']:2d} "
        f"regressed={row['regressed_from_sft']:2d} "
        f"delta={row['paired_delta']:+.3f} "
        f"p={row['mcnemar_exact_p']:.6f}"
    )

print()
print("EVAL_ROOT=", root)
print("[PASS] subset evaluation complete")
PY

cp -f "$EVAL_ROOT/summary.csv" \
    "$HOME/groot_n17_${SET_NAME}_summary.csv"

cp -f "$EVAL_ROOT/paired_comparison.csv" \
    "$HOME/groot_n17_${SET_NAME}_paired.csv"

echo
echo "EVAL_ROOT=$EVAL_ROOT"
echo "N17_FIXED100_SUBSET_EVALUATION_COMPLETE"
