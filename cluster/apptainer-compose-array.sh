#!/bin/bash

# Define variables

# PROJECT_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/mozaik-models/Rozsa_Cagnol2024" 
PROJECT_ROOT="$PWD/../mozaik-models/experanto" 
SIF_IMAGE="${SIF_IMAGE:-$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif}"
ENV_FILE=".env"
MOZAIK_ROOT="$PWD"
EXPERANTO_ROOT="$PWD/../../experanto"
# DATA_ROOT="$PWD/../data"
DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"
# DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"

echo PROJECT_ROOT: $PROJECT_ROOT        
echo SIF_IMAGE: $SIF_IMAGE
echo ENV_FILE: $ENV_FILE
echo MOZAIK_ROOT: $MOZAIK_ROOT
echo EXPERANTO_ROOT: $EXPERANTO_ROOT

# Inherit the thread counts populated by your SLURM script
# Fallback to 8 if not running under SLURM
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${OMP_NUM_THREADS:-4}

# Capture SLURM tasks, default to 4
export NTASKS=${SLURM_NTASKS:-12}

echo "Starting Mozaik Container..."
# Optional input-dataset override: only injected when BASE_PATH is set, so confs that don't set it
# produce a byte-identical apptainer argv (keeps the P1_launch golden gate green).
BASE_PATH_ARG=()
if [ -n "${BASE_PATH:-}" ]; then
  BASE_PATH_ARG=(--env "BASE_PATH=$BASE_PATH")
  echo "BASE_PATH override: $BASE_PATH"
fi
# Optional datastore redirect + workspace bind (same byte-identical pattern): only injected when set,
# so confs that don't set them keep the P1_launch golden gate green. RESULTS_DIR redirects the sim's
# datastore dir (run.py results_dir override, applied in mozaik-simulation-array.sh); WORKSPACE binds a
# host path (e.g. a ceph-ssd workspace) to /ws so RESULTS_DIR=/ws/... lands there.
RESULTS_DIR_ARG=()
if [ -n "${RESULTS_DIR:-}" ]; then
  RESULTS_DIR_ARG=(--env "RESULTS_DIR=$RESULTS_DIR")
  echo "RESULTS_DIR override: $RESULTS_DIR"
fi
WORKSPACE_ARG=()
if [ -n "${WORKSPACE:-}" ]; then
  WORKSPACE_ARG=(--bind "$WORKSPACE:/ws")
  echo "WORKSPACE bind: $WORKSPACE -> /ws"
fi
apptainer exec \
 --cleanenv \
 --env OMPI_MCA_orte_tmpdir_base=/tmp \
 --env PYTHONPATH="/mozaik:$PYTHONPATH" \
 --env TRIAL="$TRIAL" \
 --env CHUNK="$CHUNK" \
 --env CHUNK_DIR="${CHUNK_DIR:-/data/mozaik_chunk}" \
 --env PARAM_FILE="${PARAM_FILE:-param/defaults}" \
 "${BASE_PATH_ARG[@]}" \
 "${RESULTS_DIR_ARG[@]}" \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
 --env OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS \
 --env NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS \
 --env NTASKS=$NTASKS \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$MOZAIK_ROOT:/mozaik" \
 --bind "$EXPERANTO_ROOT:/experanto" \
 --bind "$DATA_ROOT:/data" \
 "${WORKSPACE_ARG[@]}" \
 "$SIF_IMAGE" \
 bash cluster/runners/mozaik-simulation-array.sh