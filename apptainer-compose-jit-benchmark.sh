#!/bin/bash

# Apptainer Compose Script for JIT Benchmark
# 
# This script runs the Mozaik JIT benchmark within an apptainer container.
# It compares JIT-enabled vs JIT-disabled simulation execution and produces
# a timing/equivalence report.
#
# Usage:
#   ./apptainer-compose-jit-benchmark.sh [model] [config] [output_dir] [duration_ms]

PROJECT_ROOT="$PWD/../mozaik-models/experanto" 
SIF_IMAGE="$PWD/../mozaik-sif/mozaik-jit.sif"
MOZAIK_ROOT="$PWD"
DATA_ROOT="$PWD/../data"

# Benchmark parameters (can override via command line)
MODEL="${1:-devtools.dummy_model.DummyModel}"
CONFIG="${2:-param/defaults}"
OUTPUT_DIR="${3:-./benchmark_results}"
DURATION="${4:-1000}"

echo "PROJECT_ROOT: $PROJECT_ROOT"        
echo "SIF_IMAGE: $SIF_IMAGE"
echo "MOZAIK_ROOT: $MOZAIK_ROOT"
echo "MODEL: $MODEL"
echo "CONFIG: $CONFIG"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DURATION: ${DURATION}ms"

# Thread configuration
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${OMP_NUM_THREADS:-4}

# MPI task count
export NTASKS=${SLURM_NTASKS:-1}

echo "Starting Mozaik JIT Benchmark Container..."

apptainer exec \
  --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env PYTHONPATH="/mozaik:$PYTHONPATH" \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
  --env OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS \
  --env NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS \
  --env NTASKS=$NTASKS \
  --bind "$PROJECT_ROOT:/project" \
  --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$DATA_ROOT:/data" \
  "$SIF_IMAGE" \
  bash /mozaik/apptainer-runners/mozaik-jit-benchmark.sh "$MODEL" "$CONFIG" "$OUTPUT_DIR" "$DURATION"

exit $?
