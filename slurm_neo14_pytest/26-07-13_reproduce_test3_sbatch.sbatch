#!/bin/bash
# Reproduce the handoff's test3 nt12 timing (docs/handoffs/26-07-03_00-16 = 16:55) under FAITHFUL
# cluster conditions, to test whether the neo14 stack is ~2x slower on the get_data path.
# 2-arm array: arm0 = neo14 refmatch SIF (neo 0.14.4), arm1 = old pre-neo14 SIF (neo 0.12.0).
# numpy is 1.26.4 in BOTH -> neo is the only get_data-relevant dependency difference.
# Node-exclusive (mem=250G), 12 ranks x 4 cores, nomultithread, sapphirerapids = handoff conditions.
# Isolated: unique run-name + unused noise_seed per arm, direct run.py (no rm -rf) -> S500 untouched.
#
# Submit from mozaik repo root:  sbatch slurm_neo14_pytest/26-07-13_reproduce_test3_sbatch.sbatch
#SBATCH --account=nix00014
#SBATCH --job-name=repro-test3-nt12
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --array=0
#SBATCH --output=./slurm_neo14_pytest/repro-test3-%A_%a.out
#SBATCH --error=./slurm_neo14_pytest/repro-test3-%A_%a.err

set -euo pipefail
module load apptainer

# Threading env matching the REAL test3 pipeline (cluster/apptainer-compose-array.sh + run_job.sh):
# OMP_NUM_THREADS only. Do NOT set OMP_PLACES/OMP_PROC_BIND here — with mpirun's own binding under
# SLURM (--cpus-per-task=4 --hint=nomultithread) they collided all ranks' threads onto the same
# cores and crippled the run. The handoff's 16:55 was produced WITHOUT them.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"                       # = mozaik repo root
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT=/mnt/vast-react/projects/neural_foundation_model
SCR="$PWD/slurm_neo14_pytest"                # inner script lives here (durable, bound to /scr)

REFMATCH="$PWD/../mozaik-sif/mozaik-opt-refmatch_2026-07-13.sif"
OLD="$PWD/../mozaik-sif/mozaik-opt.sif"
SIFS=(  "$REFMATCH"          "$OLD" )
TAGS=(  sb_neo14_refmatch    sb_neo12_old )
SEEDS=( 999993              999992 )

i=${SLURM_ARRAY_TASK_ID}
SIF="${SIFS[$i]}"; TAG="${TAGS[$i]}"; SEED="${SEEDS[$i]}"
echo "=== arm $i: $TAG  SIF=$(basename "$SIF")  on $(hostname)  $(date) ==="

SECONDS=0
apptainer exec --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env PYTHONPATH="/mozaik" \
  --env TRIAL=0 --env CHUNK=0 --env CHUNK_DIR=/data/MOZAIK/mozaik_chunk_test3 --env N_CHUNKS=1 \
  --env PARAM_FILE=param/defaults \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --env NTASKS="$SLURM_NTASKS" --env RUN_NAME="$TAG" --env NOISE_SEED="$SEED" \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$EXPERANTO_ROOT:/experanto" --bind "$DATA_ROOT:/data" --bind "$SCR:/scr" \
  "$SIF" bash /scr/26-07-13_smoke_run_sbatch.sh || echo "ARM $TAG FAILED (rc=$?)"
echo "=== ARM_${TAG}_WALL=${SECONDS}s  (handoff baseline test3 nt12 = 1015s / 16:55) ==="
