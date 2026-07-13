#!/bin/bash
# NEO-ISOLATION arm: neo14 refmatch SIF but with neo 0.12.0 overriding 0.14.4 on PYTHONPATH
# (everything else identical: same imagen/PyNN/NEST/numpy/quantities, current mozaik code).
# Faithful golden conditions (node-exclusive, 12x4, nomultithread, NO OMP_PROC_BIND). Isolated
# run-name/seed + direct run.py (no rm -rf) -> S500 untouched.
# Compare its blank per-presentation time (~expect 76s if neo is the cause) vs the neo14 arm (~150s)
# and the golden (mozaik-opt.sif, neo 0.12) = ~76s / total 16:55.
#
# Submit from mozaik repo root:  sbatch slurm_neo14_pytest/26-07-13_neo12override_test3_sbatch.sbatch
#SBATCH --account=nix00014
#SBATCH --job-name=neo12ovr-test3
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --output=./slurm_neo14_pytest/neo12ovr-test3-%j.out
#SBATCH --error=./slurm_neo14_pytest/neo12ovr-test3-%j.err

set -euo pipefail
module load apptainer

# Match the REAL test3 pipeline threading: OMP_NUM_THREADS only, NO OMP_PLACES/OMP_PROC_BIND.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT=/mnt/vast-react/projects/neural_foundation_model
SCR="$PWD/slurm_neo14_pytest"
NEO012=/mnt/vast-nhr/projects/nix00014/goirik/tmp/neo012-override   # neo 0.12.0 (pip --target)
SIF="$PWD/../mozaik-sif/mozaik-opt-refmatch_2026-07-13.sif"

TAG=sb_neo12override; SEED=999990
echo "=== $TAG  SIF=$(basename "$SIF")  neo-override=$NEO012  on $(hostname)  $(date) ==="

SECONDS=0
apptainer exec --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env PYTHONPATH="/neo012:/mozaik" \
  --env TRIAL=0 --env CHUNK=0 --env CHUNK_DIR=/data/MOZAIK/mozaik_chunk_test3 --env N_CHUNKS=1 \
  --env PARAM_FILE=param/defaults \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --env NTASKS="$SLURM_NTASKS" --env RUN_NAME="$TAG" --env NOISE_SEED="$SEED" \
  --bind "$NEO012:/neo012" \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$EXPERANTO_ROOT:/experanto" --bind "$DATA_ROOT:/data" --bind "$SCR:/scr" \
  "$SIF" bash /scr/26-07-13_smoke_run_sbatch.sh || echo "ARM $TAG FAILED (rc=$?)"
echo "=== ${TAG}_WALL=${SECONDS}s  (golden neo12 test3 nt12 = 1015s / 16:55) ==="
