#!/bin/bash
# Build the REFERENCE-MATCHING SIF (NEST at -O3, no -march=native) from the new def.
# Non-destructive: new SIF name; refuses to overwrite. Builds from a clean context dir so the
# def's vestigial `%files . /app` copies only the def (not the GB-heavy working repo).
set -euo pipefail

DEF_SRC=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik/mozaik-opt-neo14_2026-07_refmatch.def
OUT=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik-sif/mozaik-opt-refmatch_2026-07-13.sif
CTX=/mnt/vast-nhr/projects/nix00014/goirik/tmp/refmatch-build-ctx

if [ -e "$OUT" ]; then echo "REFUSE: $OUT exists (non-destructive)."; exit 1; fi

export APPTAINER_TMPDIR=/mnt/vast-nhr/projects/nix00014/goirik/tmp/apptainer-build-tmp
export APPTAINER_CACHEDIR=$APPTAINER_TMPDIR/cache
mkdir -p "$APPTAINER_CACHEDIR" "$CTX"

# Proxy for apt/pip/git/wget inside %post (GWDG cache), per the session sbatch.
export HTTP_PROXY="http://www-cache.gwdg.de:3128" HTTPS_PROXY="http://www-cache.gwdg.de:3128" FTP_PROXY="http://www-cache.gwdg.de:3128"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" ftp_proxy="$FTP_PROXY"

cp "$DEF_SRC" "$CTX/refmatch.def"
cd "$CTX"

echo "=== [$(date)] building $OUT ==="
apptainer build --fakeroot "$OUT" refmatch.def

echo "=== [$(date)] BUILD COMPLETE: $OUT ($(du -h "$OUT" | cut -f1)) ==="
echo "=== verify NEST flags in the new SIF ==="
# --cleanenv is REQUIRED: without it the container inherits the host TMPDIR (lustre, read-only
# inside the container) and OpenMPI's orte session-dir mkdir fails on `import nest`.
apptainer exec --cleanenv --env OMPI_MCA_orte_tmpdir_base=/tmp "$OUT" bash -lc '
  unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
  . /opt/mozaik/bin/activate
  /opt/mozaik/bin/nest-config --cflags | tr " " "\n" | grep -E "^-O|march|mtune" || true
  python -c "import nest,neo,numpy;print(\"neo\",neo.__version__,\"numpy\",numpy.__version__)"'
