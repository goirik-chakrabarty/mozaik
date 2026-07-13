#!/bin/bash
# NON-DESTRUCTIVE validation build: recompile NEST 3.4 at plain -O3 (NO -march=native),
# matching the upstream CSNG-MFF recipe / GitHub CI, into node-local /local scratch only.
# Nothing on VAST/lustre is touched; /local is auto-cleaned when the SLURM job ends.
# Runs INSIDE the existing SIF so it uses the SAME gsl/boost/gcc/python as production.
set -euo pipefail

SCR=/mnt/vast-nhr/projects/nix00014/goirik/tmp/nest-o3-validate
SRC=$SCR/nest-src
PREFIX=$SCR/nest-o3-install
mkdir -p "$SCR"

echo "=== [$(date)] download NEST 3.4 source ==="
cd "$SCR"
rm -rf "$SRC" nest-simulator-3.4 v3.4.tar.gz
wget -q https://github.com/nest/nest-simulator/archive/v3.4.tar.gz
tar xzf v3.4.tar.gz
mv nest-simulator-3.4 "$SRC"

echo "=== [$(date)] cmake (-O3 ONLY, no -march=native) ==="
mkdir -p "$SRC/_build"
cd "$SRC/_build"
cmake \
    -Dwith-mpi=ON \
    -Dwith-boost=ON \
    -DCMAKE_INSTALL_PREFIX:PATH="$PREFIX" \
    -Dwith-optimize='-O3' \
    -Dwith-gsl=ON \
    -Dwith-python=ON \
    "$SRC"

echo "=== [$(date)] make (-j32) ==="
make -j32
echo "=== [$(date)] make install ==="
make -j32 install

echo "=== [$(date)] verify flags ==="
"$PREFIX"/bin/nest-config --cflags | tr ' ' '\n' | grep -E '^-O|march|mtune' || true
echo "=== [$(date)] BUILD DONE -> $PREFIX ==="
