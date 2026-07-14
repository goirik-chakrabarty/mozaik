#!/bin/bash
# Inner runner for the test3 nt12 speed-reproduction sbatch. Mirrors the production runner
# (cluster/runners/mozaik-simulation-array.sh) VERBATIM for the mpirun/run.py call (plain
# `mpirun -n $NTASKS`, SLURM-bound), but with a UNIQUE run-name + unused noise_seed and NO rm -rf,
# so it never touches the live S500 datastores. Runs INSIDE the SIF.
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
. /opt/mozaik/bin/activate 2>/dev/null || true
cd /project
NTASKS=${NTASKS:-12}
RUN_NAME="${RUN_NAME:?}"; NOISE_SEED="${NOISE_SEED:?}"
echo "=== NEST flags: $(nest-config --cflags | tr ' ' '\n' | grep -E '^-O|march|mtune' | paste -sd' ') | neo $(python -c 'import neo;print(neo.__version__)') ==="
echo "=== run_name=$RUN_NAME seed=$NOISE_SEED NTASKS=$NTASKS OMP=$OMP_NUM_THREADS PLACES=${OMP_PLACES:-} BIND=${OMP_PROC_BIND:-} ==="
mpirun -n "$NTASKS" \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x NUMEXPR_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
    -x PYTHONPATH \
    python -u run.py nest "$NTASKS" "${PARAM_FILE:-param/defaults}" \
        lgn_stepcurrentsource_noise_seed "$NOISE_SEED" "$RUN_NAME"
echo "=== run.py exit: $? ==="
