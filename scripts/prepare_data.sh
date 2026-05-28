#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data}"
mkdir -p "${ROOT}"

cat <<'NOTE'
Place the licensed datasets under the data root:

  ${ROOT}/mrnet     MRNet      https://stanfordmlgroup.github.io/competitions/mrnet/
  ${ROOT}/mura      MURA       https://stanfordmlgroup.github.io/competitions/mura/
  ${ROOT}/oai       OAI        https://nda.nih.gov/oai
  ${ROOT}/vqa_rad   VQA-RAD    https://osf.io/89kps/
  ${ROOT}/slake     SLAKE      https://www.med-vqa.com/slake/

MRNet expects train/{axial,coronal,sagittal}/<case>.npy plus train-<task>.csv and the valid split.
MURA expects the official train_image_paths.csv and valid_image_paths.csv.
Each access link states the dataset license and any required data use agreement.
NOTE
