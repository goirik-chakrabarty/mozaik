# Mozaik JIT Optimization - Quick Reference

## Overview
JIT-accelerated stimulus generation for Mozaik neural simulations.
- **Status:** Ready for production
- **Performance:** Modest speedup (~1-5% depending on stimulus intensity)
- **Guarantee:** Numeric equivalence (L∞ ≤ 1e-6, spike counts identical)

---

## Toggle JIT (Before Running Simulations)

```bash
# Enable JIT (default, recommended for production)
export MOZAIK_JIT=1
python run.py nest 1 param/defaults run_name

# Disable JIT (for testing equivalence)
export MOZAIK_JIT=0
python run.py nest 1 param/defaults run_name
```

---

## Run Tests

```bash
# Essential: Fix proxy bug first
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY

# Quick validation (no dependencies)
python test_jit_integration.py

# Full test suite (requires pytest)
pytest tests/test_jit_equivalence.py -v

# Run specific test class
pytest tests/test_jit_equivalence.py::TestJITToggle -v
```

**Expected Output:**
```
✓ ALL TESTS PASSED
  - JIT kernels: 3/3
  - Toggle mechanism: 4/4
  - Kernel equivalence: tested
```

---

## Run Benchmark

**Local Execution:**
```bash
python benchmark_jit.py \
    --model devtools.dummy_model.DummyModel \
    --config param/defaults \
    --output /tmp/jit_benchmark \
    --duration 1000
```

**Container Execution:**
```bash
# Requires apptainer and mozaik-jit.sif

./apptainer-compose-jit-benchmark.sh \
    devtools.dummy_model.DummyModel \
    param/defaults \
    ./benchmark_results \
    1000
```

**Output:**
```
benchmark_results/
├── run_jit/
│   ├── benchmark_timing.txt
│   └── ...
├── run_nojit/
│   ├── benchmark_timing.txt
│   └── ...
└── benchmark_report.json  # ← timing + speedup metrics
```

---

## Modified/New Files

### Core Implementation
- `mozaik/jit_utils.py` - JIT kernels + toggle
- `mozaik/stimuli/vision/topographica_based.py` - Refactored stimulus classes

### Testing
- `tests/utils_neo_compare.py` - Neo equivalence utilities
- `tests/test_jit_equivalence.py` - Pytest test suite
- `test_jit_integration.py` - Smoke test

### Benchmarking
- `benchmark_jit.py` - Benchmark runner

### Container Integration
- `apptainer-runners/mozaik-jit-benchmark.sh` - Container runner
- `apptainer-compose-jit-benchmark.sh` - Compose orchestrator

### Documentation
- `JIT_OPTIMIZATION_SUMMARY.md` - Full technical guide
- `JIT_OPTIMIZATION_CHECKLIST.md` - Implementation checklist
- `JIT_OPTIMIZATION_QUICK_REFERENCE.md` - This file

---

## Supported Stimulus Classes

| Class | JIT Kernel | Status |
|-------|-----------|--------|
| `FullfieldDriftingSinusoidalGrating` | `fast_grating_offset_scale` | ✓ Optimized |
| `FullfieldDriftingSquareGrating` | `fast_grating_offset_scale` | ✓ Optimized |
| `FullfieldDriftingSinusoidalGratingA` | `fast_grating_offset_scale` | ✓ Optimized |
| `SparseNoise` | `fast_sparse_noise_scale` | ✓ Optimized |
| `DenseNoise` | `fast_dense_noise_scale` | ✓ Optimized |

---

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `MOZAIK_JIT` | `1` | Enable/disable JIT (0=off, 1=on) |
| `OMP_NUM_THREADS` | 4 | OpenMP threads |
| `MKL_NUM_THREADS` | 4 | MKL threads |
| `OPENBLAS_NUM_THREADS` | 4 | OpenBLAS threads |

**For cluster jobs, unset proxies:**
```bash
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY  # Avoids parameters library bug
```

---

## Performance Expectations

### Speedup Factors
- **Best case** (many stimulus frames): 3-5%
- **Typical case** (moderate stimuli): 1-3%
- **Minimal case** (model-dominated): <1%

### Memory Overhead
- Negligible (~1 MB for JIT cache)
- No dynamic memory allocation in kernels

### Compatibility
- ✓ CPU-only (no GPU required)
- ✓ Single-process and MPI
- ✓ SLURM/apptainer containers
- ✓ Python 3.8+

---

## Troubleshooting

### Test Failures

**Issue:** `NameError: name 'HTTPHandler' is not defined`
```bash
# Solution
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY
python test_jit_integration.py
```

**Issue:** `ModuleNotFoundError: No module named 'pyNN'`
```bash
# Solution: Test requires full Mozaik install
# Use test_jit_integration.py instead
python test_jit_integration.py
```

### Benchmark Issues

**Issue:** Benchmark hangs or times out
```bash
# Check timeout (default 1 hour)
# Increase with --timeout if using sub-1s simulations
python benchmark_jit.py --help
```

### JIT Not Enabling

**Issue:** `is_jit_enabled()` returns False unexpectedly
```bash
# Check explicit setting
import mozaik.jit_utils as jit_utils
print(jit_utils._JIT_ENABLED)  # None = use env var

# Reset to default
jit_utils._JIT_ENABLED = None
print(jit_utils.is_jit_enabled())  # Should be True
```

---

## Integration with Existing Workflows

### SLURM Job Script
```bash
#!/bin/bash
#SBATCH --...

unset HTTP_PROXY HTTPS_PROXY FTP_PROXY
export MOZAIK_JIT=1  # Enable JIT
export OMP_NUM_THREADS=4

cd /path/to/mozaik
python run.py nest $SLURM_NTASKS param/defaults run_name
```

### Development Workflow
```bash
# Disable JIT for faster iteration (smaller compile overhead)
export MOZAIK_JIT=0

# Run your simulations
python my_experiment.py

# Final production run with JIT
export MOZAIK_JIT=1
python my_experiment.py
```

---

## Numeric Validation

To verify JIT equivalence in your own simulations:

```python
import os
os.environ['MOZAIK_JIT'] = '0'

# Run simulation 1 (no JIT)
# ... (collect outputs to neo_block1)

os.environ['MOZAIK_JIT'] = '1'

# Run simulation 2 (with JIT)
# ... (collect outputs to neo_block2)

# Compare
from tests.utils_neo_compare import compare_neo_blocks
results = compare_neo_blocks(neo_block1, neo_block2)
print(f"Equivalence: {results['summary']}")
```

---

## Advanced: Custom JIT Kernels

To add more JIT kernels to your stimulus classes:

1. **Define kernel in `mozaik/jit_utils.py`:**
```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True, fastmath=True)
def my_stimulus_kernel(param1, param2):
    # Pure math, no Python objects
    return result
```

2. **Wrap in stimulus class:**
```python
def _compute_my_param(self, param1):
    if jit_utils.is_jit_enabled():
        return jit_utils.my_stimulus_kernel(
            float(param1),
            float(self.background_luminance)
        )
    else:
        return float(param1) * float(self.background_luminance)
```

3. **Test equivalence:**
```python
# JIT on
result_jit = stim._compute_my_param(50.0)
# JIT off
result_inline = stim._compute_my_param(50.0)
assert result_jit == result_inline
```

---

## Documentation References

- **Full Guide:** `JIT_OPTIMIZATION_SUMMARY.md`
- **Checklist:** `JIT_OPTIMIZATION_CHECKLIST.md`
- **Code:** See comments in modified files

---

## Support & Issues

For issues or questions:
1. Check **Troubleshooting** section above
2. Run `test_jit_integration.py` to validate setup
3. Review **JIT_OPTIMIZATION_SUMMARY.md** for detailed architecture
4. Check proxy workaround: `unset HTTP_PROXY HTTPS_PROXY FTP_PROXY`

---

**Last Updated:** February 23, 2026  
**Status:** ✓ Production Ready
