#!/bin/bash
# Correctness gate for the quantities registry-memo patch (lever B): run the 30/30 full_model
# reference tests inside the neopatch SIF (neo O(1) baked) WITH the quantities patch bind-mounted on
# top — the exact patch stack that will ship. Must stay 30/30 green (spikes + voltages).
#SBATCH --account=nix00014
#SBATCH --job-name=qpatch-gate
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=./slurm_neo14_pytest/qpatch-gate-%j.out
#SBATCH --error=./slurm_neo14_pytest/qpatch-gate-%j.err

set -uo pipefail
module load apptainer
SIF=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt-neopatch_2026-07-13.sif
MZ=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik
QPATCH=/mnt/vast-nhr/projects/nix00014/goirik/tmp/neo-patch/quantities-0.16.4/registry.py
QDST=/opt/mozaik/lib/python3.10/site-packages/quantities/registry.py

echo "=== stack: neo $(apptainer exec --cleanenv $SIF bash -lc '. /opt/mozaik/bin/activate 2>/dev/null; python -c "import neo;print(neo.__version__)"') + quantities memo patch ==="
apptainer exec --cleanenv --bind "$QPATCH:$QDST" $SIF bash -lc '. /opt/mozaik/bin/activate 2>/dev/null; python -c "import quantities,inspect,quantities.registry as r; print(\"PERF PATCH present:\", \"PERF PATCH\" in inspect.getsource(r))"'

run_pytest () {
  apptainer exec --cleanenv --env PYTHONPATH=/mozaik --env OMP_NUM_THREADS=8 \
    --env OMPI_MCA_orte_tmpdir_base=/tmp --bind $MZ:/mozaik --bind "$QPATCH:$QDST" $SIF \
    bash -c 'unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy; cd /mozaik; python -m pytest '"$1"' -p no:randomly -v'
}

echo ""; echo "########## test_models.py :: VogelsAbbott + LSV1M_tiny + LSV1M_tiny_2024_LGN ##########"
run_pytest 'tests/full_model/test_models.py -m model -k "Tiny or VogelsAbbott2005"' 2>&1 | tail -50

echo ""; echo "########## test_models_stepcurrentmodule.py ##########"
run_pytest 'tests/full_model/test_models_stepcurrentmodule.py -m model' 2>&1 | tail -30
