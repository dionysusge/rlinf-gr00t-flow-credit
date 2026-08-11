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

run_mode() {
  local run_mode="$1"
  local -a overrides=(
    "policy_decorator.mode=${run_mode}"
    "runner.logger.log_path=${RESULT_ROOT}"
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
