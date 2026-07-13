#!/bin/bash
# Re-profile the sim with neopatch SIF (neo O(1) baked) PLUS the candidate quantities registry-memo
# patch bind-mounted. Same perf1 chunk / nt12 / OMP=4 as 26-07-13_prof_patched_sbatch.sbatch so the
# cumulative pstats are directly comparable — proves the quantities-hash cost collapses.
#SBATCH --account=nix00014
#SBATCH --job-name=prof-qpatch-perf1
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --time=00:40:00
#SBATCH --output=./slurm_neo14_pytest/prof-qpatch-%j.out
#SBATCH --error=./slurm_neo14_pytest/prof-qpatch-%j.err

set -euo pipefail
module load apptainer
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"
MOZAIK_ROOT="$PWD"
PROJECT_ROOT="$PWD/../mozaik-models/experanto"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT=/mnt/vast-react/projects/neural_foundation_model
SCR="$PWD/slurm_neo14_pytest"
SIF="$PWD/../mozaik-sif/mozaik-opt-neopatch_2026-07-13.sif"
QPATCH=/mnt/vast-nhr/projects/nix00014/goirik/tmp/neo-patch/quantities-0.16.4/registry.py
QDST=/opt/mozaik/lib/python3.10/site-packages/quantities/registry.py

echo "=== profile qpatch sim on perf1, nt12, on $(hostname) $(date) ==="
SECONDS=0
apptainer exec --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp --env PYTHONPATH="/mozaik" \
  --env TRIAL=0 --env CHUNK=0 --env CHUNK_DIR=/data/MOZAIK/mozaik_chunk_perf1 --env N_CHUNKS=1 \
  --env PARAM_FILE=param/defaults \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" --env NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  --env VECLIB_MAXIMUM_THREADS="$VECLIB_MAXIMUM_THREADS" \
  --bind "$PROJECT_ROOT:/project" --bind "$MOZAIK_ROOT:/mozaik" --bind "$EXPERANTO_ROOT:/experanto" \
  --bind "$DATA_ROOT:/data" --bind "$SCR:/scr" \
  --bind "$QPATCH:$QDST" \
  "$SIF" bash -c '
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
    . /opt/mozaik/bin/activate 2>/dev/null || true
    echo "=== neo $(python -c "import neo;print(neo.__version__)") quantities $(python -c "import quantities;print(quantities.__version__)") ==="
    cd /project
    mpirun -n 12 -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x NUMEXPR_NUM_THREADS -x VECLIB_MAXIMUM_THREADS -x PYTHONPATH \
      python -u /scr/26-07-14_prof_run_qpatch.py
  ' || echo "FAILED rc=$?"
echo "=== prof_qpatch_WALL=${SECONDS}s ==="
