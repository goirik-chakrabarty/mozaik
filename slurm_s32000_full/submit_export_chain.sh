#!/bin/bash
# Submit the S32000-full export as a CHAIN of 4 chunk-range jobs (append mode), each dependent on the
# previous (afterok) so they append into one spikes.npy sequentially. 4 ranges x 16 chunks (~20-25h each,
# comfortably < 48h even if export runs 2x slower than the pilot's ~15 MB/s). Run from the mozaik repo root
# AFTER the sim array is fully COMPLETED and all 64 datastores are present on /ws:
#   bash slurm_s32000_full/submit_export_chain.sh
# Optionally pass a sim array job id to gate the FIRST range on it: submit_export_chain.sh <SIM_JOBID>
set -euo pipefail
SBATCH=slurm_s32000_full/export_range_cephssd.sbatch
RANGES=("0 16" "16 32" "32 48" "48 64")
SIM_DEP="${1:-}"

dep=""
[ -n "$SIM_DEP" ] && dep="--dependency=afterok:$SIM_DEP"
prev=""
for r in "${RANGES[@]}"; do
  cs=${r% *}; ce=${r#* }
  d="$dep"
  [ -n "$prev" ] && d="--dependency=afterok:$prev"
  jid=$(sbatch --parsable $d --export=ALL,CHUNK_START=$cs,CHUNK_END=$ce "$SBATCH")
  echo "submitted export chunks [$cs,$ce) -> job $jid ${d:+($d)}"
  prev=$jid
done
echo "Export chain submitted. Final job: $prev (writes /ws/S32000_full/export/trial0)."
