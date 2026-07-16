#!/bin/bash
# S32000 PILOT sim on ceph-ssd (simulate final-run storage circumstances: input read + datastore
# write both on the ceph-ssd workspace, NOT project vast). Single non-colliding task trial7_chunk0,
# noise_seed=7000, qpatch SIF, 12 ranks. Datastore redirected via run.py `results_dir` override.
# Isolated workspace -> nothing touches S500. Submit from mozaik repo root:
#   sbatch slurm_s32000_pilot/26-07-14_pilot_sim_cephssd.sbatch
#SBATCH --account=nix00014
#SBATCH --job-name=s32k-pilot-sim
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_s32000_pilot/pilot-sim-%j.out
#SBATCH --error=./slurm_s32000_pilot/pilot-sim-%j.err

set -euo pipefail
module load apptainer
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"                       # = mozaik repo root
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
SCR="$PWD/slurm_s32000_pilot"
SIF="$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif"
WS=/mnt/ceph-ssd/workspaces/ws/nix00014/u18196-mozaik_s32000

echo "=== S32000 pilot sim on $(hostname) $(date) — storage=ceph-ssd ($WS) ==="
SECONDS=0
apptainer exec --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp --env PYTHONPATH="/mozaik" --env NTASKS="$SLURM_NTASKS" \
  --env TRIAL=7 --env CHUNK=0 --env N_CHUNKS=1 \
  --env CHUNK_DIR=/ws/S32000_pilot/chunks \
  --env BASE_PATH=/ws/S32000_pilot/input_screen \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$EXPERANTO_ROOT:/experanto" --bind "$WS:/ws" --bind "$SCR:/scr" \
  "$SIF" bash /scr/pilot_run_cephssd.sh || echo "PILOT SIM FAILED (rc=$?)"
echo "=== PILOT_SIM_WALL=${SECONDS}s ==="
echo "=== datastore on ceph-ssd: ==="
ls -ld "$WS"/S32000_pilot/datastores/SelfSustainedPushPull_trial7_chunk0* 2>/dev/null || echo "(no datastore produced)"
du -sh "$WS"/S32000_pilot/datastores/SelfSustainedPushPull_trial7_chunk0* 2>/dev/null || true
