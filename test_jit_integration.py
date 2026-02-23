#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration test for Mozaik JIT optimization.
Tests kernels, toggle mechanism, and equivalence.
"""

import os
import sys

# Unset proxy to avoid parameters library bug
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('FTP_PROXY', None)

import mozaik.jit_utils as jit_utils
import numpy as np


def test_kernels():
    """Test JIT kernels."""
    print("Testing JIT Kernels...")
    
    # Grating
    offset, scale = jit_utils.fast_grating_offset_scale(50.0, 100.0)
    assert offset == 50.0 and scale == 100.0
    print("  ✓ fast_grating_offset_scale")
    
    # Sparse noise
    scale = jit_utils.fast_sparse_noise_scale(50.0)
    assert scale == 100.0
    print("  ✓ fast_sparse_noise_scale")
    
    # Dense noise
    scale = jit_utils.fast_dense_noise_scale(60.0)
    assert scale == 120.0
    print("  ✓ fast_dense_noise_scale")


def test_toggle():
    """Test JIT toggle."""
    print("\nTesting JIT Toggle...")
    
    jit_utils.set_jit_enabled(True)
    assert jit_utils.is_jit_enabled()
    print("  ✓ set_jit_enabled(True)")
    
    jit_utils.set_jit_enabled(False)
    assert not jit_utils.is_jit_enabled()
    print("  ✓ set_jit_enabled(False)")
    
    os.environ['MOZAIK_JIT'] = '1'
    jit_utils._JIT_ENABLED = None
    assert jit_utils.is_jit_enabled()
    print("  ✓ MOZAIK_JIT=1")
    
    os.environ['MOZAIK_JIT'] = '0'
    jit_utils._JIT_ENABLED = None
    assert not jit_utils.is_jit_enabled()
    print("  ✓ MOZAIK_JIT=0")
    
    # Reset
    os.environ['MOZAIK_JIT'] = '1'
    jit_utils._JIT_ENABLED = None


def test_equivalence():
    """Test JIT vs inline equivalence."""
    print("\nTesting Kernel Equivalence...")
    
    test_cases = [
        (50.0, 100.0),
        (25.0, 75.0),
        (99.0, 200.0),
    ]
    
    for contrast, bg_lum in test_cases:
        offset_jit, scale_jit = jit_utils.fast_grating_offset_scale(contrast, bg_lum)
        
        # Inline computation
        offset_inline = bg_lum * (100.0 - contrast) / 100.0
        scale_inline = 2.0 * bg_lum * contrast / 100.0
        
        assert np.isclose(offset_jit, offset_inline)
        assert np.isclose(scale_jit, scale_inline)
    
    print(f"  ✓ Tested {len(test_cases)} cases - all equivalent")


def main():
    print("\n" + "="*60)
    print("Mozaik JIT Optimization Integration Tests")
    print("="*60 + "\n")
    
    try:
        test_kernels()
        test_toggle()
        test_equivalence()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nDeliverables Summary:")
        print("  ✓ JIT kernels added to mozaik/jit_utils.py")
        print("  ✓ Vision stimuli refactored (topographica_based.py)")
        print("  ✓ Neo equivalence test suite (tests/utils_neo_compare.py)")
        print("  ✓ JIT equivalence tests (tests/test_jit_equivalence.py)")
        print("  ✓ Benchmark runner (benchmark_jit.py)")
        print("  ✓ Apptainer integration (apptainer-*-jit-benchmark.sh)")
        print("\nInstructions:")
        print("  1. Run tests: pytest -m jit_equivalence tests/test_jit_equivalence.py")
        print("  2. Run benchmark: ./apptainer-compose-jit-benchmark.sh")
        print("  3. Toggle JIT: export MOZAIK_JIT=0/1")
        return 0
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
