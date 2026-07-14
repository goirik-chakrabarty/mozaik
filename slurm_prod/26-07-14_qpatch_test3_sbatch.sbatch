#!/bin/bash
# PRODUCTION faithful test3 nt12 timing with the BAKED qpatch SIF (neo O(1) _contains + quantities
# registry-memo, BOTH baked in — self-contained, gate-verified 30/30; no runtime bind-mounts).
# Golden conditions (node-exclusive, 12x4, nomultithread, plain mpirun, OMP_NUM_THREADS only). Goal:
# total wall BELOW golden 1015s / 16:55 (expect ~741s). Isolated run-name/seed, direct run.py, no rm -rf.
#SBATCH --account=nix00014
#SBATCH --job-name=qpatch-test3-nt12
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --output=./slurm_prod/qpatch-test3-%j.out
#SBATCH --error=./slurm_prod/qpatch-test3-%j.err

set -euo pipefail
module load apptainer
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT=/mnt/vast-react/projects/neural_foundation_model
SCR="$PWD/slurm_prod"
SIF="$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif"   # baked: neo _contains + quantities memo (self-contained)

echo "=== qpatch test3 nt12 on $(hostname) $(date) === (golden=1015s; neopatch-only=913s)"
SECONDS=0
apptainer exec --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp --env PYTHONPATH="/mozaik" \
  --env TRIAL=0 --env CHUNK=0 --env CHUNK_DIR=/data/MOZAIK/mozaik_chunk_test3 --env N_CHUNKS=1 --env PARAM_FILE=param/defaults \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --env NTASKS="$SLURM_NTASKS" --env RUN_NAME=sb_qpatch --env NOISE_SEED=999985 \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" --bind "$EXPERANTO_ROOT:/experanto" \
  --bind "$DATA_ROOT:/data" --bind "$SCR:/scr" \
  "$SIF" bash /scr/26-07-13_smoke_run_sbatch.sh || echo "FAILED rc=$?"
echo "=== sb_qpatch_WALL=${SECONDS}s  (golden test3 nt12 = 1015s / 16:55; neopatch-only ~913s) ==="
