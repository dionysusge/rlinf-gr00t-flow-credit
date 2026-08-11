#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${PD_RESULT_ROOT:-/data/Wayne/gzw/rlinf_gr00t_n17/results/policy_decorator_gr00t}"
if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "Result directory does not exist: ${RESULT_ROOT}" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${RESULT_ROOT}/policy_decorator_evidence_${TIMESTAMP}.tar.gz"

tar \
  --exclude='wandb' \
  --exclude='tensorboard' \
  --exclude='checkpoints' \
  --exclude='*.tar.gz' \
  -czf "${ARCHIVE}" \
  -C "${RESULT_ROOT}" .

echo "Created ${ARCHIVE}"
echo "Set INCLUDE_CHECKPOINTS manually and use tar if a model checkpoint is needed."
