#!/bin/bash
# Inner runner for the S32000-full EXPORT (runs INSIDE the qpatch SIF). Exports a CHUNK RANGE of trial 0
# in APPEND mode (export.py: is_resume = CHUNK_START>0) so ranges chain into one spikes.npy per trial.
# Env: TRIAL, N_CHUNKS, CHUNK_START, CHUNK_END, DATASTORE_PREFIX, OUTPUT_PREFIX, CHUNK_DIR, BATCH_SIZE.
set -uo pipefail
. /opt/mozaik/bin/activate 2>/dev/null || true
cd /project
echo "=== neo $(python -c 'import neo;print(neo.__version__)') | TRIAL=$TRIAL chunks [$CHUNK_START,$CHUNK_END) of $N_CHUNKS ==="
echo "=== DATASTORE_PREFIX=$DATASTORE_PREFIX OUTPUT_PREFIX=$OUTPUT_PREFIX CHUNK_DIR=$CHUNK_DIR ==="
python -u export.py "$TRIAL" --n-chunks "$N_CHUNKS" \
    --chunk-start "$CHUNK_START" --chunk-end "$CHUNK_END" --batch-size "${BATCH_SIZE:-1}"
echo "=== export.py exit: $? ==="
