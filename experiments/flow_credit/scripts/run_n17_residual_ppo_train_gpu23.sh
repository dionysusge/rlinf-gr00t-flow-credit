#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"
CONFIG_NAME=libero_spatial_residual_ppo_gr00t_n1d7_h200_gpu23

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
export RLINF_FORCE_LOCAL_RAY=1
# Keep this path short: Ray embeds a long session name below it and Linux
# AF_UNIX socket paths are limited to 107 bytes.
export RAY_TMPDIR=/mnt/models/gzw/raytmp/e1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
export PYTHONPATH="$RLINF:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset MUJOCO_EGL_DEVICE_ID 2>/dev/null || true
unset RAY_ADDRESS 2>/dev/null || true

E0_ROOT=$(cat "$BULK/logs/n17_residual_e0.latest" 2>/dev/null || true)
ALLOW_UNVERIFIED_E0=${ALLOW_UNVERIFIED_E0:-0}
E0_VERIFIED_AT_LAUNCH=0
if [[ -n "$E0_ROOT" && -f "$E0_ROOT/E0_PASS" ]]; then
    E0_VERIFIED_AT_LAUNCH=1
fi
if [[ "$ALLOW_UNVERIFIED_E0" != "1" && "$E0_VERIFIED_AT_LAUNCH" != "1" ]]; then
    echo "E1 blocked: seeded E0 has not passed." >&2
    echo "For an explicitly concurrent launch, set ALLOW_UNVERIFIED_E0=1." >&2
    exit 2
fi

mkdir -p "$RAY_TMPDIR" "$BULK/runs" "$BULK/logs" "$BULK/wandb"
python "$RLINF/experiments/flow_credit/analysis/validate_ray_tmpdir.py" "$RAY_TMPDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="n17_residual_ppo_a01_gpu23_${STAMP}"
RUN_NAME="Residual-PPO-0.1-GR00T-N1.7-LIBERO-Spatial-H200x2-${STAMP}"
RUN_DIR="$BULK/runs/$RUN_ID"
export RUN_ID RUN_NAME RUN_DIR E0_ROOT ALLOW_UNVERIFIED_E0 E0_VERIFIED_AT_LAUNCH
export WANDB_RUN_ID="$RUN_ID"
export WANDB_NAME="$RUN_NAME"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$BULK/logs/n17_residual_ppo.latest"

cp "$RLINF/examples/embodiment/config/${CONFIG_NAME}.yaml" "$RUN_DIR/launch_config.yaml"
git -C "$RLINF" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$RLINF" status --short > "$RUN_DIR/git_status.txt"
printf '%s\n' "$E0_ROOT" > "$RUN_DIR/e0_evaluation_root.txt"
if [[ -f "$E0_ROOT/E0_REPEATABILITY.json" ]]; then
    cp "$E0_ROOT/E0_REPEATABILITY.json" "$RUN_DIR/e0_repeatability.json"
fi
if [[ "$E0_VERIFIED_AT_LAUNCH" != "1" ]]; then
    printf '%s\n' \
        "E1 was launched concurrently before seeded E0 completed." \
        "Treat this run as provisional until E0_PASS exists." \
        > "$RUN_DIR/E0_UNVERIFIED_AT_LAUNCH.txt"
fi

python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    --cfg job \
    --resolve \
    runner.logger.log_path="$RUN_DIR" \
    runner.logger.experiment_name="$RUN_NAME" \
    > "$RUN_DIR/resolved_config.yaml"

python - "$RUN_DIR/run_manifest.json" "$RLINF" "$CONFIG_NAME" <<'PY'
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

manifest_path = Path(sys.argv[1])
repo = Path(sys.argv[2])
config_name = sys.argv[3]

def capture(command):
    try:
        return subprocess.run(
            command, text=True, check=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"

try:
    import torch
    torch_version = torch.__version__
except ImportError:
    torch_version = "unavailable"

manifest = {
    "started_at": datetime.now().astimezone().isoformat(),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python_version": sys.version,
    "torch_version": torch_version,
    "git_commit": capture(["git", "-C", str(repo), "rev-parse", "HEAD"]),
    "git_branch": capture(["git", "-C", str(repo), "branch", "--show-current"]),
    "gpu": capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader",
        ]
    ),
    "config_name": config_name,
    "gr00t_checkpoint_path": "/data/Wayne/gzw/rlinf_gr00t_n17/models/GR00T-N1.7-LIBERO/libero_spatial",
    "actor_seed": 1234,
    "rollout_seed": 1234,
    "env_seed": 0,
    "e0_root": os.environ.get("E0_ROOT"),
    "e0_verified_at_launch": os.environ.get("E0_VERIFIED_AT_LAUNCH") == "1",
    "allow_unverified_e0": os.environ.get("ALLOW_UNVERIFIED_E0") == "1",
    "run_id": os.environ.get("RUN_ID"),
    "run_name": os.environ.get("RUN_NAME"),
    "command": (
        "python examples/embodiment/train_embodied_agent.py "
        f"--config-name {config_name} "
        f"runner.logger.log_path={os.environ.get('RUN_DIR')} "
        f"runner.logger.experiment_name={os.environ.get('RUN_NAME')}"
    ),
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "============================================================"
echo "Residual PPO E1"
echo "run:                 $RUN_NAME"
echo "GPUs:                2,3"
echo "base GR00T:          frozen"
echo "residual bound:      0.1"
echo "actor/critic LR:     1e-4 / 1e-4"
echo "transitions/update:  4096"
echo "checkpoint steps:    30,60,90,120,150"
echo "checkpoint approx:   123K,246K,369K,492K,614K transitions"
echo "E0 root:             $E0_ROOT"
echo "E0 verified launch:  $E0_VERIFIED_AT_LAUNCH"
if [[ "$E0_VERIFIED_AT_LAUNCH" != "1" ]]; then
    echo "E0 status:           PROVISIONAL concurrent launch"
fi
echo "rollout seed:         1234 (rank-offset per rollout worker)"
echo "run dir:             $RUN_DIR"
echo "============================================================"

cd "$RLINF"
python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$RUN_DIR" \
    runner.logger.experiment_name="$RUN_NAME" \
    2>&1 | tee "$RUN_DIR/training.log"

python experiments/flow_credit/analysis/log_evidence_to_wandb.py \
    --kind training \
    --root "$RUN_DIR" \
    --name "$RUN_NAME" \
    --run-id "$RUN_ID"

echo "N17_RESIDUAL_PPO_E1_COMPLETE"
