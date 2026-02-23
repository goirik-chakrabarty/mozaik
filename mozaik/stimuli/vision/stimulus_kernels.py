# -*- coding: utf-8 -*-
"""
Refactored vision stimulus classes using Numba JIT kernels for math-heavy operations.

This module provides JIT-optimized wrappers for vision stimulus generation.
Each stimulus class separates:
  - Math kernels (pure NumPy, JIT-compiled)
  - Python wrappers (handling metadata, Neo routing, RNG synchronization)

The JIT kernels are called only when MOZAIK_JIT is enabled; otherwise,
the wrappers fall back to inline math (no performance difference for correctness).
"""

from mozaik.stimuli.vision.visual_stimulus import VisualStimulus
from mozaik.tools.mozaik_parametrized import SNumber
from mozaik.tools.units import cpd
from numpy import pi
from quantities import Hz, rad, degrees, ms, dimensionless
import numpy as np
import mozaik.jit_utils as jit_utils


class JITAcceleratedVisualStimulus(VisualStimulus):
    """
    Base class for JIT-accelerated vision stimuli.
    
    Provides common infrastructure for wrappers to call JIT kernels
    conditionally based on MOZAIK_JIT environment variable.
    """
    
    def __init__(self, **params):
        VisualStimulus.__init__(self, **params)
        self._jit_enabled = jit_utils.is_jit_enabled()
    
    def get_jit_enabled(self):
        """Query whether JIT is enabled for this stimulus instance."""
        return self._jit_enabled
    
    def compute_grating_params(self, contrast):
        """
        Compute offset and scale for grating stimuli with given contrast.
        
        Parameters
        ----------
        contrast : float
            Contrast percentage (0-100).
        
        Returns
        -------
        tuple (offset, scale)
            Imagen-compatible offset and scale parameters.
        """
        if self._jit_enabled:
            return jit_utils.fast_grating_offset_scale(contrast, float(self.background_luminance))
        else:
            # Inline fallback for correctness validation
            offset = float(self.background_luminance) * (100.0 - contrast) / 100.0
            scale = 2.0 * float(self.background_luminance) * contrast / 100.0
            return offset, scale
    
    def compute_sparse_noise_scale(self):
        """Compute scale for sparse noise stimulus."""
        if self._jit_enabled:
            return jit_utils.fast_sparse_noise_scale(float(self.background_luminance))
        else:
            return 2.0 * float(self.background_luminance)
    
    def compute_dense_noise_scale(self):
        """Compute scale for dense noise stimulus."""
        if self._jit_enabled:
            return jit_utils.fast_dense_noise_scale(float(self.background_luminance))
        else:
            return 2.0 * float(self.background_luminance)
