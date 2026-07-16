#!/bin/bash
# Inner runner for the S32000 pilot EXPORT (runs INSIDE the qpatch SIF). Reads the pilot datastore and
# writes the Experanto export on the ceph-ssd workspace (bound at /ws). This validates the VIDEO export
# path (movie_frame_duration_ms=35, untested at S500) and measures export size.
# TRIAL/N_CHUNKS/CHUNK_DIR/OUTPUT_PREFIX/DATASTORE_PREFIX/BATCH_SIZE come from --env.
set -uo pipefail
. /opt/mozaik/bin/activate 2>/dev/null || true
cd /project
echo "=== neo $(python -c 'import neo;print(neo.__version__)') | TRIAL=$TRIAL N_CHUNKS=$N_CHUNKS ==="
echo "=== DATASTORE_PREFIX=$DATASTORE_PREFIX OUTPUT_PREFIX=$OUTPUT_PREFIX CHUNK_DIR=$CHUNK_DIR ==="
python -u export.py "$TRIAL" --n-chunks "$N_CHUNKS" --batch-size "${BATCH_SIZE:-1}"
echo "=== export.py exit: $? ==="
