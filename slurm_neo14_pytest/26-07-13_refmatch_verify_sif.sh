#!/bin/bash
# Full reference-test verification INSIDE the new refmatch SIF (NEST -O3, no -march).
# Confirms ALL previously-failing voltage tests now pass: VogelsAbbott2005, LSV1M_tiny,
# LSV1M_tiny_2024_LGN (test_models.py) + the stepcurrentmodule models (test_models_stepcurrentmodule.py).
# We DELIBERATELY exclude the MozaikModelsSmoke clone tests (the upstream mpi_seed ParameterSet
# mismatch the user asked to ignore).
set -uo pipefail
SIF=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt-refmatch_2026-07-13.sif
MZ=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

echo "=== NEST flags baked into the new SIF (expect -O3, NO march/mtune) ==="
apptainer exec --cleanenv $SIF bash -lc '. /opt/mozaik/bin/activate; nest-config --cflags | tr " " "\n" | grep -E "^-O|march|mtune" || true'

run_pytest () {
  apptainer exec --cleanenv --env PYTHONPATH=/mozaik --env OMP_NUM_THREADS=4 \
    --env OMPI_MCA_orte_tmpdir_base=/tmp --bind $MZ:/mozaik $SIF \
    bash -c 'unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy; cd /mozaik; python -m pytest '"$1"' -p no:randomly -v'
}

echo ""; echo "########## test_models.py :: VogelsAbbott + LSV1M_tiny + LSV1M_tiny_2024_LGN ##########"
run_pytest 'tests/full_model/test_models.py -m model -k "Tiny or VogelsAbbott2005"' 2>&1 | tail -45

echo ""; echo "########## test_models_stepcurrentmodule.py ##########"
run_pytest 'tests/full_model/test_models_stepcurrentmodule.py -m model' 2>&1 | tail -25
