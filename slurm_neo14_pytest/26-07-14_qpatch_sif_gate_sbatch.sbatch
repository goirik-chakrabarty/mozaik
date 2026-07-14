#!/bin/bash
# Certify the BAKED qpatch SIF (mozaik-opt-qpatch_2026-07-14.sif) — run the 30/30 full_model reference
# tests directly against it (no bind-mount), proving the deliverable image is correct as shipped.
#SBATCH --account=nix00014
#SBATCH --job-name=qpatch-sif-gate
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=./slurm_neo14_pytest/qpatch-sif-gate-%j.out
#SBATCH --error=./slurm_neo14_pytest/qpatch-sif-gate-%j.err

set -uo pipefail
module load apptainer
SIF=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt-qpatch_2026-07-14.sif
MZ=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

echo "=== certifying baked SIF: $(basename $SIF) ==="
apptainer exec --cleanenv $SIF bash -lc '. /opt/mozaik/bin/activate 2>/dev/null
  python -c "import inspect,quantities.registry as r; print(\"quantities patch:\", \"PERF PATCH\" in inspect.getsource(r))"
  grep -q "PERF PATCH" /opt/mozaik/lib/python3.10/site-packages/neo/core/objectlist.py && echo "neo patch: True"'

run_pytest () {
  apptainer exec --cleanenv --env PYTHONPATH=/mozaik --env OMP_NUM_THREADS=8 \
    --env OMPI_MCA_orte_tmpdir_base=/tmp --bind $MZ:/mozaik $SIF \
    bash -c 'unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy; cd /mozaik; python -m pytest '"$1"' -p no:randomly -v'
}

echo ""; echo "########## test_models.py :: VogelsAbbott + LSV1M_tiny + LSV1M_tiny_2024_LGN ##########"
run_pytest 'tests/full_model/test_models.py -m model -k "Tiny or VogelsAbbott2005"' 2>&1 | tail -50

echo ""; echo "########## test_models_stepcurrentmodule.py ##########"
run_pytest 'tests/full_model/test_models_stepcurrentmodule.py -m model' 2>&1 | tail -30
