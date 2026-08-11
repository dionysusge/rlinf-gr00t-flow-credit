#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  zero_smoke|random_smoke|train|eval|all) ;;
  *)
    echo "Usage: $0 {zero_smoke|random_smoke|train|eval|all}" >&2
    exit 2
    ;;
esac

PROJECT="${PD_PROJECT_ROOT:-/data/Wayne/gzw/rlinf_gr00t_n17}"
ACTIVATE_SCRIPT="${PD_ACTIVATE_SCRIPT:-${PROJECT}/scripts/activate_rlinf.sh}"
if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "Policy Decorator environment script not found: ${ACTIVATE_SCRIPT}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${PD_GPUS:-2,3}"
export EMBODIED_PATH="${EMBODIED_PATH:-${REPO_ROOT}/examples/embodiment}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR=1

RESULT_ROOT="${PD_RESULT_ROOT:-/data/Wayne/gzw/rlinf_gr00t_n17/results/policy_decorator_gr00t}"
ENTRY="examples/embodiment/train_policy_decorator_gr00t.py"
BASE_MODEL_PATH="${PD_MODEL_PATH:-${PROJECT}/models/GR00T-N1.7-LIBERO/libero_spatial}"
if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "GR00T checkpoint directory not found: ${BASE_MODEL_PATH}" >&2
  exit 2
fi

if [[ -n "${PD_BACKBONE_MODEL_PATH:-}" ]]; then
  BACKBONE_MODEL_PATH="${PD_BACKBONE_MODEL_PATH}"
elif [[ -d "${PROJECT}/models/Cosmos-Reason2-2B" ]]; then
  BACKBONE_MODEL_PATH="${PROJECT}/models/Cosmos-Reason2-2B"
elif [[ -d "/data/Wayne/gzw/gr00t_n1_7/models/Cosmos-Reason2-2B" ]]; then
  BACKBONE_MODEL_PATH="/data/Wayne/gzw/gr00t_n1_7/models/Cosmos-Reason2-2B"
else
  echo "Cosmos-Reason2-2B was not found in either known server location." >&2
  exit 2
fi
echo "Policy Decorator GR00T checkpoint: ${BASE_MODEL_PATH}"
echo "Policy Decorator Cosmos backbone:  ${BACKBONE_MODEL_PATH}"

run_mode() {
  local run_mode="$1"
  local -a overrides=(
    "policy_decorator.mode=${run_mode}"
    "runner.logger.log_path=${RESULT_ROOT}"
    "actor.model.model_path=${BASE_MODEL_PATH}"
    "actor.model.backbone_model_path=${BACKBONE_MODEL_PATH}"
  )
  if [[ "${run_mode}" =~ ^(train|eval)$ && -n "${PD_RESUME:-}" ]]; then
    overrides+=("policy_decorator.resume_path=${PD_RESUME}")
  fi
  python "${ENTRY}" "${overrides[@]}"
}

if [[ "${MODE}" == "all" ]]; then
  run_mode zero_smoke
  run_mode random_smoke
  run_mode train
else
  run_mode "${MODE}"
fi
