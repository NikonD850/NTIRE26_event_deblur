#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python basicsr/utils/raw_event_to_teid_voxel21.py \
  --dataroot /home/fdw/data/event_based_motion_deblur/HighREV/test \
  --backend cpu \
  --num-workers 0 \
  --output-root /home/fdw/data/event_based_motion_deblur/HighREV/test \
  --output-dir-name voxel21 \
  --bins 7 \
  --tl 5 \
  --tm 1 \
  --ts 0 \
  --overwrite

