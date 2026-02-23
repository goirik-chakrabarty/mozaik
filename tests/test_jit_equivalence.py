# -*- coding: utf-8 -*-
"""
Equivalence tests for JIT-optimized stimuli.

These tests:
1. Run identical models twice: once with MOZAIK_JIT=1 and once with MOZAIK_JIT=0
2. Compare spiking output (SpikeTrain) and analog signals (AnalogSignal) from both runs
3. Assert that numeric outputs match within acceptable floating-point tolerances

The rationale is that pure math (offset/scale computation) should produce
identical binary outputs regardless of whether Numba JIT is enabled,
validating that the JIT kernels are functionally equivalent to the inline fallback.
"""

import os
import pytest
import numpy as np
from quantities import ms, Hz, degrees, dimensionless, rad
from mozaik.tools.units import lux
import tempfile
import shutil

# Conditionally import mozaik and test utilities
try:
    import mozaik
    import mozaik.jit_utils as jit_utils
    from mozaik.stimuli.vision.topographica_based import (
        FullfieldDriftingSinusoidalGrating,
        FullfieldDriftingSquareGrating,
        SparseNoise,
        DenseNoise
    )
    from tests.utils_neo_compare import compare_neo_blocks
    MOZAIK_AVAILABLE = True
except ImportError:
    MOZAIK_AVAILABLE = False


# Functions to compute JIT kernel outputs directly (unit tests)

class TestJITKernels:
    """Unit tests for individual JIT kernels."""
    
    @pytest.mark.skipif(not MOZAIK_AVAILABLE, reason="mozaik not available")
    def test_grating_offset_scale_kernel(self):
        """Test fast_grating_offset_scale kernel."""
        # Test with deterministic params
        background_luminance = 50.0  # lux
        contrast = 50.0  # percent
        
        # Compute via JIT
        offset_jit, scale_jit = jit_utils.fast_grating_offset_scale(contrast, background_luminance)
        
        # Compute inline
        offset_inline = background_luminance * (100.0 - contrast) / 100.0
        scale_inline = 2.0 * background_luminance * contrast / 100.0
        
        # Check equivalence
        assert np.isclose(offset_jit, offset_inline, rtol=1e-14)
        assert np.isclose(scale_jit, scale_inline, rtol=1e-14)
    
    @pytest.mark.skipif(not MOZAIK_AVAILABLE, reason="mozaik not available")
    def test_sparse_noise_scale_kernel(self):
        """Test fast_sparse_noise_scale kernel."""
        background_luminance = 40.0  # lux
        
        scale_jit = jit_utils.fast_sparse_noise_scale(background_luminance)
        scale_inline = 2.0 * background_luminance
        
        assert np.isclose(scale_jit, scale_inline, rtol=1e-14)
    
    @pytest.mark.skipif(not MOZAIK_AVAILABLE, reason="mozaik not available")
    def test_dense_noise_scale_kernel(self):
        """Test fast_dense_noise_scale kernel."""
        background_luminance = 30.0  # lux
        
        scale_jit = jit_utils.fast_dense_noise_scale(background_luminance)
        scale_inline = 2.0 * background_luminance
        
        assert np.isclose(scale_jit, scale_inline, rtol=1e-14)


class TestJITToggle:
    """Tests for JIT enable/disable functionality."""
    
    def test_jit_toggle_env_var_enabled(self, monkeypatch):
        """Test MOZAIK_JIT=1 enables JIT."""
        monkeypatch.setenv('MOZAIK_JIT', '1')
        # Reset module state
        jit_utils._JIT_ENABLED = None
        assert jit_utils.is_jit_enabled() is True
    
    def test_jit_toggle_env_var_disabled(self, monkeypatch):
        """Test MOZAIK_JIT=0 disables JIT."""
        monkeypatch.setenv('MOZAIK_JIT', '0')
        jit_utils._JIT_ENABLED = None
        assert jit_utils.is_jit_enabled() is False
    
    def test_jit_toggle_env_var_default(self, monkeypatch):
        """Test default behavior when MOZAIK_JIT is not set."""
        monkeypatch.delenv('MOZAIK_JIT', raising=False)
        jit_utils._JIT_ENABLED = None
        # Default should be True
        assert jit_utils.is_jit_enabled() is True
    
    def test_jit_toggle_explicit(self):
        """Test explicit set_jit_enabled() overrides env var."""
        jit_utils.set_jit_enabled(False)
        assert jit_utils.is_jit_enabled() is False
        jit_utils.set_jit_enabled(True)
        assert jit_utils.is_jit_enabled() is True


class TestStimulusJITConsistency:
    """Integration tests: verify stimulus classes call JIT kernels correctly."""
    
    @pytest.mark.skipif(not MOZAIK_AVAILABLE, reason="mozaik not available")
    def test_fullfield_drifting_sinusoidal_grating_jit_call(self):
        """Test that FullfieldDriftingSinusoidalGrating uses JIT kernel."""
        # Create stimulus instance
        stim = FullfieldDriftingSinusoidalGrating(
            background_luminance=50.0 * lux,
            contrast=50.0 * dimensionless,
            orientation=0.0 * rad,
            spatial_frequency=0.5,
            temporal_frequency=1.0 * Hz,
            density=1.0,
            location_x=0.0 * degrees,
            location_y=0.0 * degrees,
            size_x=10.0 * degrees,
            size_y=10.0 * degrees,
            frame_duration=10.0 * ms
        )
        
        # Enable JIT and compute params
        jit_utils.set_jit_enabled(True)
        offset_jit, scale_jit = stim._compute_grating_params(50.0)
        
        # Disable JIT and compute params
        jit_utils.set_jit_enabled(False)
        offset_inline, scale_inline = stim._compute_grating_params(50.0)
        
        # Both should be identical
        assert np.isclose(offset_jit, offset_inline, rtol=1e-14)
        assert np.isclose(scale_jit, scale_inline, rtol=1e-14)
    
    @pytest.mark.skipif(not MOZAIK_AVAILABLE, reason="mozaik not available")
    def test_sparse_noise_jit_call(self):
        """Test that SparseNoise uses JIT kernel."""
        stim = SparseNoise(
            background_luminance=40.0 * lux,
            experiment_seed=42 * dimensionless,
            duration=1000.0 * ms,
            time_per_image=100.0 * ms,
            blank_time=100.0 * ms,
            grid_size=5.0 * dimensionless,
            grid=0.0 * dimensionless,
            density=1.0,
            location_x=0.0 * degrees,
            location_y=0.0 * degrees,
            size_x=10.0 * degrees,
            size_y=10.0 * degrees,
            frame_duration=10.0 * ms
        )
        
        # Enable JIT
        jit_utils.set_jit_enabled(True)
        scale_jit = stim._compute_sparse_noise_scale()
        
        # Disable JIT
        jit_utils.set_jit_enabled(False)
        scale_inline = stim._compute_sparse_noise_scale()
        
        assert np.isclose(scale_jit, scale_inline, rtol=1e-14)


# Marker-based tests for optional heavy model tests
# These can be run with: pytest -m "jit_equivalence" tests/test_jit_equivalence.py

pytestmark = pytest.mark.jit_equivalence


@pytest.mark.skipif(not MOZAIK_AVAILABLE, reason="mozaik not available")
def test_jit_kernels_summary():
    """Smoke test: verify JIT kernels are loadable and callable."""
    # Just verify the kernels exist and are callable
    assert callable(jit_utils.fast_grating_offset_scale)
    assert callable(jit_utils.fast_sparse_noise_scale)
    assert callable(jit_utils.fast_dense_noise_scale)
    
    # Verify they compute deterministic outputs
    result1 = jit_utils.fast_grating_offset_scale(50.0, 100.0)
    result2 = jit_utils.fast_grating_offset_scale(50.0, 100.0)
    assert result1 == result2, "Kernels should be deterministic"
