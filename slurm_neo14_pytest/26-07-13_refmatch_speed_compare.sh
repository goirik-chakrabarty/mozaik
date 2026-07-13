#!/bin/bash
# A/B speed comparison: test3 @ 12 ranks x 4 OMP, refmatch (-O3) vs production (-march=native),
# run SEQUENTIALLY on the same node for a fair timing. Isolated: unique run-name + unused noise_seed
# per SIF, run.py invoked directly (no rm -rf) -> never touches the live S500 datastores.
set -uo pipefail
SCR=/mnt/lustre-grete/tmp/u18196/claude-897072/-mnt-vast-nhr-projects-nix00014-goirik-MOZAIK-new/2744b2e7-6da2-42ea-b081-772eefc8b2aa/scratchpad
MZ=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik
PROJECT_ROOT=$MZ/../mozaik-models/experanto
EXPERANTO_ROOT=$MZ/../../experanto
DATA_ROOT=/mnt/vast-react/projects/neural_foundation_model
REFSIF=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt-refmatch_2026-07-13.sif
PRODSIF=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt_2026-07-10.sif

run_one () {
  local SIF="$1" TAG="$2" SEED="$3"
  echo "############################################################"
  echo "### $TAG  |  SIF=$(basename "$SIF")  |  12 ranks x OMP=4  |  seed=$SEED"
  echo "############################################################"
  local t0 t1
  t0=$(date +%s)
  apptainer exec --cleanenv \
    --env OMPI_MCA_orte_tmpdir_base=/tmp \
    --env PYTHONPATH="/mozaik:${PYTHONPATH:-}" \
    --env TRIAL=0 --env CHUNK=0 --env CHUNK_DIR=/data/MOZAIK/mozaik_chunk_test3 --env N_CHUNKS=1 \
    --env PARAM_FILE=param/defaults \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env NUMEXPR_NUM_THREADS=4 \
    --env NTASKS=12 --env RUN_NAME="$TAG" --env NOISE_SEED="$SEED" \
    --bind "$PROJECT_ROOT:/project" --bind "$MZ:/mozaik" --bind "$EXPERANTO_ROOT:/experanto" \
    --bind "$DATA_ROOT:/data" --bind "$SCR:/scr" \
    "$SIF" bash /scr/smoke_run.sh > "$SCR/speed_${TAG}.log" 2>&1
  t1=$(date +%s)
  echo "WALL_${TAG}=$((t1 - t0)) s"
  # NEST-only portion: sum reported 'of which Ns was simulation time'
  local simsum
  simsum=$(grep -aoE 'of which [0-9]+ ?s was simulation' "$SCR/speed_${TAG}.log" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
  echo "SIMSUM_${TAG}=${simsum} s (NEST sim.run portion)"
}

run_one "$REFSIF"  refmatch_o3      999997
run_one "$PRODSIF" production_native 999996

echo ""
echo "================= SUMMARY ================="
grep -aE '^WALL_|^SIMSUM_' "$SCR/speed_compare.log" 2>/dev/null || true
echo "(also see speed_refmatch_o3.log / speed_production_native.log)"
