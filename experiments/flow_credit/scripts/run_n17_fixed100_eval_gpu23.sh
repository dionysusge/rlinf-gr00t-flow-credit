#!/usr/bin/env bash
set -euo pipefail

# Fixed-100 eval launches 100 LIBERO subprocesses.
ulimit -n 65535 2>/dev/null || true

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
export RAY_TMPDIR="/tmp/r17e_apps"
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"

mkdir -p \
    "$BULK/tmp/ray-fixed-eval" \
    "$BULK/evaluations" \
    "$BULK/logs"

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

test -n "$RUN_DIR" || {
    echo "[FAIL] FP32Master run directory not found"
    exit 1
}

CKPT_ROOT=$(
    find "$RUN_DIR" \
        -type d \
        -path '*/checkpoints' \
        -print \
    | head -1
)

test -d "$CKPT_ROOT" || {
    echo "[FAIL] checkpoint root not found"
    exit 1
}

for step in 200 400 600 650
do
    test -d "$CKPT_ROOT/global_step_${step}/actor" || {
        echo "[FAIL] missing checkpoint: global_step_${step}"
        exit 1
    }
done

STAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT="$BULK/evaluations/n17_fixed100_${STAMP}"

mkdir -p "$EVAL_ROOT"

echo "$EVAL_ROOT" \
    > "$BULK/logs/n17_fixed100_eval.latest"

echo "$CKPT_ROOT" \
    > "$EVAL_ROOT/checkpoint_root.txt"

declare -a LABELS=(
    "sft_base"
    "step200"
    "step400"
    "step600"
    "step650"
)

declare -a RESUME_DIRS=(
    "null"
    "$CKPT_ROOT/global_step_200"
    "$CKPT_ROOT/global_step_400"
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

    echo
    echo "============================================================"
    echo "fixed evaluation: $label"
    echo "resume_dir:       $resume_dir"
    echo "output_dir:       $output_dir"
    echo "============================================================"

    if [[ "$resume_dir" == "null" ]]; then
        resume_override="runner.resume_dir=null"
    else
        resume_override="runner.resume_dir=$resume_dir"
    fi

    set +e

    EVAL_LABEL="$label" \
    python "$ENTRY" \
        --config-path "$EMBODIED_PATH/config" \
        --config-name "$CONFIG_NAME" \
        "$resume_override" \
        "runner.logger.log_path=$output_dir" \
        "runner.logger.experiment_name=fixed100-$label" \
        2>&1 | tee "$log_file"

    rc=${PIPESTATUS[0]}

    set -e

    echo "$rc" > "$output_dir/exit_code.txt"

    # 当前串行评测独占本次 Ray 集群。
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

    echo "[PASS] $label"
done

python - "$EVAL_ROOT" <<'PY'
from pathlib import Path
import csv
import json
import sys

root = Path(sys.argv[1])

label_order = [
    "sft_base",
    "step200",
    "step400",
    "step600",
    "step650",
]

rows = []

for label in label_order:
    path = root / label / "metrics.json"

    if not path.is_file():
        raise SystemExit(
            f"[FAIL] metrics missing: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    metrics = payload["metrics"]

    trajectories = int(metrics["num_trajectories"])

    if trajectories != 100:
        raise SystemExit(
            f"[FAIL] {label}: expected 100 trajectories, "
            f"got {trajectories}"
        )

    rows.append(
        {
            "label": label,
            "resume_dir": payload.get("resume_dir"),
            "num_trajectories": trajectories,
            "success_once": float(
                metrics["success_once"]
            ),
            "return": float(metrics["return"]),
            "reward": float(metrics["reward"]),
            "episode_len": float(
                metrics["episode_len"]
            ),
        }
    )

summary_path = root / "summary.csv"

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)

best = max(
    rows,
    key=lambda row: (
        row["success_once"],
        row["reward"],
        -row["episode_len"],
    ),
)

best_path = root / "best_model.json"

best_path.write_text(
    json.dumps(
        best,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("===== fixed 100-trajectory summary =====")
print(
    f"{'label':10s} "
    f"{'success':>9s} "
    f"{'reward':>10s} "
    f"{'ep_len':>9s} "
    f"{'n':>5s}"
)

for row in rows:
    print(
        f"{row['label']:10s} "
        f"{row['success_once']:9.4f} "
        f"{row['reward']:10.6f} "
        f"{row['episode_len']:9.2f} "
        f"{row['num_trajectories']:5d}"
    )

print()
print("best label:      ", best["label"])
print("best success:    ", best["success_once"])
print("best reward:     ", best["reward"])
print("best episode len:", best["episode_len"])
print()
print("[PASS] all fixed evaluations complete")
PY

cp -f \
    "$EVAL_ROOT/summary.csv" \
    "$HOME/groot_n17_fixed100_summary.csv"

cp -f \
    "$EVAL_ROOT/best_model.json" \
    "$HOME/groot_n17_fixed100_best_model.json"

echo
echo "EVAL_ROOT=$EVAL_ROOT"
echo "SUMMARY=$EVAL_ROOT/summary.csv"
echo "SFTP_SUMMARY=$HOME/groot_n17_fixed100_summary.csv"
echo
echo "N17_FIXED100_EVALUATION_COMPLETE"
