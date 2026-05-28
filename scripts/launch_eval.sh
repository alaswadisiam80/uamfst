#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.gin}"
CHECKPOINT="${2:-runs/mrnet_main/model.ckpt}"

python -m uamfst.cellarmaster.scoring \
  --gin_file "${EXPERIMENT}" \
  --checkpoint "${CHECKPOINT}"
