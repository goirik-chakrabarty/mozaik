#!/bin/bash
# S32000-full: materialize the Experanto input_screen from subset_S32000_allvid.json (41,723 records =
# 31,706 img + 10,017 unique vid) onto the ceph-ssd workspace. One-off prep (NOT the gated pipeline).
# materialize_subset_screen.py --selection-json copies pixels from the source pool (absolute session_paths
# under /mnt/vast-react/...) to <ws>/S32000_full/input_screen/screen/{meta,data}. --out refuses overwrite.
# Single-threaded, I/O/metadata-bound (~83k small-file ops) -> shared partition, few cores. Submit from mozaik root:
#   sbatch slurm_s32000_full/26-07-14_materialize_cephssd.sbatch
#SBATCH --account=nix00014
#SBATCH --job-name=s32k-materialize
#SBATCH --partition=standard96:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_s32000_full/materialize-%j.out
#SBATCH --error=./slurm_s32000_full/materialize-%j.err

set -euo pipefail
module load apptainer

cd "$SLURM_SUBMIT_DIR"                        # = mozaik repo root
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
SIF="$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif"
WS=/mnt/ceph-ssd/workspaces/ws/nix00014/u18196-mozaik_s32000
SRC=/mnt/vast-react/projects/neural_foundation_model     # source pool (absolute session_paths)

echo "=== S32000-full materialize on $(hostname) $(date) ==="
SECONDS=0
apptainer exec --cleanenv \
  --bind "$PROJECT_ROOT:/project" \
  --bind "$SRC:$SRC" \
  --bind "$WS:/ws" \
  "$SIF" bash -c '
    . /opt/mozaik/bin/activate 2>/dev/null || true
    cd /project
    python -u materialize_subset_screen.py \
      --selection-json /ws/S32000_full/subset_S32000_allvid.json \
      --out /ws/S32000_full/input_screen
  ' || echo "MATERIALIZE FAILED (rc=$?)"
echo "=== MATERIALIZE_WALL=${SECONDS}s ==="
echo "=== output ==="
ls -la "$WS"/S32000_full/input_screen/screen 2>/dev/null || echo "(no output)"
echo "meta count:"; ls "$WS"/S32000_full/input_screen/screen/meta 2>/dev/null | wc -l
echo "data count:"; ls "$WS"/S32000_full/input_screen/screen/data 2>/dev/null | wc -l
du -sh "$WS"/S32000_full/input_screen 2>/dev/null || true
