#!/bin/bash
# Inner runner for the S32000 pilot sim (runs INSIDE the qpatch SIF). Reads input + writes datastore
# on the ceph-ssd workspace (bound at /ws). TRIAL/CHUNK/CHUNK_DIR/BASE_PATH/N_CHUNKS come from --env.
# results_dir is passed as a QUOTED python literal because run.py eval()s override values (mozaik/cli.py:18).
set -uo pipefail
. /opt/mozaik/bin/activate 2>/dev/null || true
cd /project
echo "=== neo $(python -c 'import neo;print(neo.__version__)') | BASE_PATH=$BASE_PATH CHUNK_DIR=$CHUNK_DIR TRIAL=$TRIAL CHUNK=$CHUNK N_CHUNKS=$N_CHUNKS ==="
mpirun -n "${NTASKS:-12}" \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x NUMEXPR_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
    -x PYTHONPATH -x TRIAL -x CHUNK -x N_CHUNKS -x CHUNK_DIR -x BASE_PATH \
    python -u run.py nest "${NTASKS:-12}" param/defaults \
        results_dir "'/ws/S32000_pilot/datastores/'" \
        lgn_stepcurrentsource_noise_seed 7000 \
        trial7_chunk0
echo "=== run.py exit: $? ==="
