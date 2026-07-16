#!/bin/bash
# S32000 PILOT export on ceph-ssd. Reads the pilot datastore (trial7_chunk0, seed 7000) and writes the
# Experanto export on the same ceph-ssd workspace (mirrors final-run I/O). Validates the VIDEO export path
# (movie_frame_duration_ms=35, untested at S500) and measures export size. Single process, qpatch SIF.
# Binds workspace -> /ws. DATASTORE_PREFIX/OUTPUT_PREFIX/CHUNK_DIR all point at /ws. Submit from mozaik root:
#   sbatch slurm_s32000_pilot/26-07-14_pilot_export_cephssd.sbatch
#
# NOTE (pilot-only): TRIAL=0, NOT 7. The sim ran under TRIAL=7 (datastore named trial7, seed 7000) for
# collision-avoidance, but the pilot chunk 7_0.json is trial-0 content (records carry trial=0, identical
# to 0_0.json). export.py filters segments by `stim_params["trial"] == TRIAL`, so it must run as TRIAL=0
# to match. A symlink datastores/...trial0_chunk0...:0 -> the real ...trial7...:7000 dir lets export locate
# the data (created 2026-07-14). The FULL run has no such mismatch (generate_chunks stamps each record's
# trial to match its file/sim trial), so it exports with matching TRIAL and needs no symlink.
#SBATCH --account=nix00014
#SBATCH --job-name=s32k-pilot-export
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=./slurm_s32000_pilot/pilot-export-%j.out
#SBATCH --error=./slurm_s32000_pilot/pilot-export-%j.err

set -euo pipefail
module load apptainer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8

cd "$SLURM_SUBMIT_DIR"                       # = mozaik repo root
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
SCR="$PWD/slurm_s32000_pilot"
SIF="$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif"
WS=/mnt/ceph-ssd/workspaces/ws/nix00014/u18196-mozaik_s32000

echo "=== S32000 pilot export on $(hostname) $(date) — storage=ceph-ssd ($WS) ==="
SECONDS=0
apptainer exec --cleanenv \
  --env PYTHONPATH="/mozaik" \
  --env TRIAL=0 --env N_CHUNKS=1 --env BATCH_SIZE=1 \
  --env CHUNK_DIR=/ws/S32000_pilot/chunks \
  --env OUTPUT_PREFIX=/ws/S32000_pilot/export/trial \
  --env DATASTORE_PREFIX=/ws/S32000_pilot/datastores \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$EXPERANTO_ROOT:/experanto" --bind "$WS:/ws" --bind "$SCR:/scr" \
  "$SIF" bash /scr/pilot_export_cephssd.sh || echo "PILOT EXPORT FAILED (rc=$?)"
echo "=== PILOT_EXPORT_WALL=${SECONDS}s ==="
echo "=== export on ceph-ssd: ==="
ls -lR "$WS"/S32000_pilot/export/trial0 2>/dev/null | head -60 || echo "(no export produced)"
du -sh "$WS"/S32000_pilot/export/trial0 2>/dev/null || true
du -sh "$WS"/S32000_pilot/export/trial0/responses "$WS"/S32000_pilot/export/trial0/screen 2>/dev/null || true
