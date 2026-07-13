#!/bin/bash
# Build the neopatch SIF: refmatch (-O3) + neo _contains O(1) perf patch baked in.
set -euo pipefail
DEF_SRC=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik/mozaik-opt-neo14_2026-07_neopatch.def
PATCH=/mnt/vast-nhr/projects/nix00014/goirik/tmp/neo-patch
OUT=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt-neopatch_2026-07-13.sif
CTX=/mnt/vast-nhr/projects/nix00014/goirik/tmp/neopatch-build-ctx
if [ -e "$OUT" ]; then echo "REFUSE: $OUT exists (non-destructive)."; exit 1; fi
export APPTAINER_TMPDIR=/mnt/vast-nhr/projects/nix00014/goirik/tmp/apptainer-build-tmp
export APPTAINER_CACHEDIR=$APPTAINER_TMPDIR/cache
mkdir -p "$APPTAINER_CACHEDIR" "$CTX"
export HTTP_PROXY="http://www-cache.gwdg.de:3128" HTTPS_PROXY="http://www-cache.gwdg.de:3128" FTP_PROXY="http://www-cache.gwdg.de:3128"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" ftp_proxy="$FTP_PROXY"
# build context: def + patched neo files (copied to /app by %files, applied over neo in %post)
cp "$DEF_SRC" "$CTX/neopatch.def"
cp "$PATCH/objectlist.py" "$PATCH/spiketrainlist.py" "$CTX/"
cd "$CTX"
echo "=== [$(date)] building $OUT ==="
apptainer build --fakeroot "$OUT" neopatch.def
echo "=== [$(date)] BUILD COMPLETE: $OUT ($(du -h "$OUT" | cut -f1)) ==="
apptainer exec --cleanenv --env OMPI_MCA_orte_tmpdir_base=/tmp "$OUT" bash -lc '
  unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy; . /opt/mozaik/bin/activate
  nest-config --cflags | tr " " "\n" | grep -E "^-O|march" | paste -sd" "
  python -c "import inspect,neo,neo.core.objectlist as m; print(\"neo\",neo.__version__, \"| patch:\", \"PERF PATCH\" in inspect.getsource(m.ObjectList._contains))"'
