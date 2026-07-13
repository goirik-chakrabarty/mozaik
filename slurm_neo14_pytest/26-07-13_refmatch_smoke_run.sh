#!/bin/bash
# Runs INSIDE the refmatch SIF. Isolated test3-style sim: same experanto experiment + test3 chunk,
# but run.py is invoked DIRECTLY with a UNIQUE run-name + unused noise_seed, so it writes a fresh
# datastore and NEVER rm -rf's / collides with the live S500 datastores. NON-DESTRUCTIVE.
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
. /opt/mozaik/bin/activate 2>/dev/null || true

cd /project
NTASKS=${NTASKS:-4}
RUN_NAME="${RUN_NAME:-refmatch_test3_smoke}"
NOISE_SEED="${NOISE_SEED:-999999}"   # S500 uses 0-5/1000-1005/2000-2005 only -> 99999x are free

echo "=== NEST build flags (expect -O3, no march) ==="
nest-config --cflags | tr " " "\n" | grep -E "^-O|march|mtune" || true
echo "=== sim: run_name=$RUN_NAME noise_seed=$NOISE_SEED TRIAL=$TRIAL CHUNK=$CHUNK CHUNK_DIR=$CHUNK_DIR NTASKS=$NTASKS ==="

mpirun --oversubscribe \
    -n "$NTASKS" \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x PYTHONPATH \
    python -u run.py nest "$NTASKS" "${PARAM_FILE:-param/defaults}" \
        lgn_stepcurrentsource_noise_seed "$NOISE_SEED" "$RUN_NAME"
rc=$?
echo "=== run.py exit code: $rc ==="
echo "=== resulting datastore dir ==="
ls -d /project/SelfSustainedPushPull_${RUN_NAME}_* 2>&1
exit $rc
