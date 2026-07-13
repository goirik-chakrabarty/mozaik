#!/bin/bash
# Run the LSV1M_tiny reference test using the scratch -O3 NEST (no -march=native) instead of the
# SIF's -march=native NEST. If -march=native was the cause of the voltage failures, test_voltages
# should now PASS (spikes were already passing — kept as a control).
set -uo pipefail

NEST_O3=/mnt/vast-nhr/projects/nix00014/goirik/tmp/nest-o3-validate/nest-o3-install
SP="$NEST_O3/lib/python3.10/site-packages"

# Prepend the scratch NEST ahead of the SIF's /opt/mozaik NEST; keep the bind-mounted mozaik pkg (/mozaik).
export PYTHONPATH="$SP:/mozaik:${PYTHONPATH:-}"
export PATH="$NEST_O3/bin:$PATH"
export LD_LIBRARY_PATH="$NEST_O3/lib:$NEST_O3/lib64:${LD_LIBRARY_PATH:-}"

echo "=== sanity: which NEST + its build flags ==="
which nest-config
nest-config --cflags | tr ' ' '\n' | grep -E '^-O|march|mtune' || true
python -c "import nest, os; print('nest module ->', os.path.dirname(nest.__file__))"
echo "(expect: nest module path under nest-o3-install; flags -O3 and NO -march/-mtune)"
echo ""

cd /mozaik
echo "=== pytest LSV1M_tiny spikes+voltages (scratch -O3 NEST) ==="
python -m pytest "tests/full_model/test_models.py::TestLSV1MTiny" -m model -p no:randomly -v 2>&1 | tail -40
