#!/bin/bash
# S32000-full EXPORT of one CHUNK RANGE (trial 0) on ceph-ssd. Standalone (chaining via --dependency is
# simplest outside the gated submit.sh, and mirrors the validated pilot export launcher). CHUNK_START/
# CHUNK_END arrive via `sbatch --export=ALL,CHUNK_START=..,CHUNK_END=..`. Ranges chain (afterok) into one
# spikes.npy because CHUNK_START>0 => export.py append mode. Submit via submit_export_chain.sh.
#SBATCH --account=nix00014
#SBATCH --job-name=s32k-export
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_s32000_full/export-%j.out
#SBATCH --error=./slurm_s32000_full/export-%j.err

set -euo pipefail
module load apptainer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8

cd "$SLURM_SUBMIT_DIR"
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
SCR="$PWD/slurm_s32000_full"
SIF="$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif"
WS=/mnt/ceph-ssd/workspaces/ws/nix00014/u18196-mozaik_s32000

: "${CHUNK_START:?set via --export=ALL,CHUNK_START=..}"
: "${CHUNK_END:?set via --export=ALL,CHUNK_END=..}"

echo "=== S32000-full export chunks [$CHUNK_START,$CHUNK_END) on $(hostname) $(date) ==="
SECONDS=0
apptainer exec --cleanenv \
  --env PYTHONPATH="/mozaik" \
  --env TRIAL=0 --env N_CHUNKS=64 \
  --env CHUNK_START="$CHUNK_START" --env CHUNK_END="$CHUNK_END" --env BATCH_SIZE=1 \
  --env CHUNK_DIR=/ws/S32000_full/chunks \
  --env OUTPUT_PREFIX=/ws/S32000_full/export/trial \
  --env DATASTORE_PREFIX=/ws/S32000_full/datastores \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$EXPERANTO_ROOT:/experanto" --bind "$WS:/ws" --bind "$SCR:/scr" \
  "$SIF" bash /scr/export_range_cephssd.sh || echo "EXPORT RANGE FAILED (rc=$?)"
echo "=== EXPORT_RANGE_WALL=${SECONDS}s (chunks $CHUNK_START..$CHUNK_END) ==="
du -sh "$WS"/S32000_full/export/trial0 2>/dev/null || true
