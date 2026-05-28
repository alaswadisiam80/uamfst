#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.gin}"
GPUS="${2:-1}"

torchrun --standalone --nproc_per_node="${GPUS}" \
  -m uamfst.cellarmaster.vintage --gin_file "${EXPERIMENT}"
