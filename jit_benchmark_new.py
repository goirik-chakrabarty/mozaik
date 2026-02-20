import time
import numpy as np
import sys
import os

# ==========================================
# 1. IMPORTS & SETUP
# ==========================================

print("--- Importing Modules ---")

# Import Original (Slow) Components
try:
    from mozaik.connectors.vision import gabor as slow_gabor
    from mozaik.connectors.vision import gauss as slow_gauss
    from mozaik.connectors.vision import V1CorrelationBasedConnectivity
    print("SUCCESS: Imported original functions from mozaik.connectors.vision")
except ImportError:
    print("ERROR: Could not import from mozaik.connectors.vision.")
    print("       Ensure you are running this from the mozaik root directory.")
    sys.exit(1)

# Import New JIT (Fast) Components
try:
    from mozaik.jit_utils import fast_gabor, fast_gauss, fast_stf_view_update, fast_integral_vectorized
    JIT_AVAILABLE = True
    print("SUCCESS: Imported 'fast' JIT functions from mozaik.jit_utils")
except ImportError:
    JIT_AVAILABLE = False
    print("WARNING: Could not import 'mozaik.jit_utils'.")
    print("         Benchmarks will only run the 'Slow' baseline versions.")

# Helper for RF View Benchmark (Original Logic)
def slow_stf_view(view_array, kernel_contrast, background_lum, contrast_resp, luminance_resp, 
                  kernel_luminance, mean_val, current_idx, update_factor, duration):
    # Logic extracted from CellWithReceptiveField.view()
    scaled_input = view_array.reshape(-1) / background_lum
    contrast_tc = np.dot(kernel_contrast, scaled_input)
    luminance_tc = kernel_luminance * mean_val
    
    for j in range(update_factor):
        start = current_idx + j
        if start + duration <= len(contrast_resp):
            contrast_resp[start : start+duration] += contrast_tc
            luminance_resp[start : start+duration] += luminance_tc

# ==========================================
# 2. BENCHMARK RUNNERS
# ==========================================

def run_geometric_benchmark(n_calls=200000):
    print(f"\n=== 1. Geometric Kernels (Setup Phase) [{n_calls} calls] ===")
    
    # Data Setup
    x1, y1 = 0.5, 0.5
    x2_list = np.random.rand(n_calls)
    y2_list = np.random.rand(n_calls)
    orientation = 0.45
    freq = 5.0
    phase = 0.0
    size = 0.2
    ar = 1.0

    # --- Gabor ---
    print("\n[Gabor Function]")
    t0 = time.time()
    for i in range(n_calls):
        slow_gabor(x1, y1, x2_list[i], y2_list[i], orientation, freq, phase, size, ar)
    t_slow = time.time() - t0
    print(f"  Baseline (Original): {t_slow:.4f} s")

    if JIT_AVAILABLE:
        # Warmup
        fast_gabor(x1, y1, 0.1, 0.1, orientation, freq, phase, size, ar)
        
        t0 = time.time()
        for i in range(n_calls):
            fast_gabor(x1, y1, x2_list[i], y2_list[i], orientation, freq, phase, size, ar)
        t_fast = time.time() - t0
        print(f"  Numba JIT          : {t_fast:.4f} s")
        print(f"  Speedup            : {t_slow / t_fast:.1f}x")

    # --- Gauss ---
    print("\n[Gauss Function]")
    t0 = time.time()
    for i in range(n_calls):
        slow_gauss(x1, y1, x2_list[i], y2_list[i], orientation, size, ar)
    t_slow = time.time() - t0
    print(f"  Baseline (Original): {t_slow:.4f} s")

    if JIT_AVAILABLE:
        # Warmup
        fast_gauss(x1, y1, 0.1, 0.1, orientation, size, ar)
        
        t0 = time.time()
        for i in range(n_calls):
            fast_gauss(x1, y1, x2_list[i], y2_list[i], orientation, size, ar)
        t_fast = time.time() - t0
        print(f"  Numba JIT          : {t_fast:.4f} s")
        print(f"  Speedup            : {t_slow / t_fast:.1f}x")


def run_rf_update_benchmark(spatial_dim=30, duration=50, n_steps=5000):
    print(f"\n=== 2. Receptive Field Update (Simulation Phase) [{n_steps} steps] ===")
    
    view_array = np.random.rand(spatial_dim, spatial_dim)
    n_pixels = spatial_dim * spatial_dim
    # Contrast kernel: (Duration, Pixels) or (Pixels, Duration) depending on implementation
    # We follow the dot product logic: dot(kernel, input) -> (Duration,)
    # So kernel must be (Duration, Pixels)
    kernel_contrast = np.random.rand(duration, n_pixels) 
    kernel_luminance = np.random.rand(duration)
    
    background_lum = 0.5
    contrast_resp = np.zeros(n_steps + duration + 10)
    luminance_resp = np.zeros(n_steps + duration + 10)
    mean_val = 0.4
    update_factor = 1
    
    # Baseline
    t0 = time.time()
    for i in range(n_steps):
        slow_stf_view(view_array, kernel_contrast, background_lum, contrast_resp, luminance_resp,
                      kernel_luminance, mean_val, i, update_factor, duration)
    t_slow = time.time() - t0
    print(f"  Baseline (Original): {t_slow:.4f} s")
    
    # JIT
    if JIT_AVAILABLE:
        contrast_resp[:] = 0
        luminance_resp[:] = 0
        
        # Warmup
        fast_stf_view_update(view_array, kernel_contrast, background_lum, contrast_resp, luminance_resp,
                             kernel_luminance, mean_val, 0, update_factor, duration)
        
        t0 = time.time()
        for i in range(n_steps):
            fast_stf_view_update(view_array, kernel_contrast, background_lum, contrast_resp, luminance_resp,
                                 kernel_luminance, mean_val, i, update_factor, duration)
        t_fast = time.time() - t0
        print(f"  Numba JIT          : {t_fast:.4f} s")
        print(f"  Speedup            : {t_slow / t_fast:.1f}x")


def run_connectivity_integral_benchmark(n_samples=50000):
    print(f"\n=== 3. Connectivity Integrals (Setup Phase) [{n_samples} pairs] ===")
    print("Testing the full analytical integral calculation for V1 connections.")
    
    # Generate Dummy Data Arrays
    K1 = np.random.rand(n_samples)
    w1_x = np.random.rand(n_samples) * 0.5
    w1_y = np.random.rand(n_samples) * 0.5
    p1_x = np.random.rand(n_samples) * 5.0
    p1_y = np.random.rand(n_samples) * 5.0
    or1 = np.random.rand(n_samples) * np.pi
    f1 = np.random.rand(n_samples) * 5.0
    ph1 = np.random.rand(n_samples) * np.pi
    
    K2 = np.random.rand(n_samples)
    w2_x = np.random.rand(n_samples) * 0.5
    w2_y = np.random.rand(n_samples) * 0.5
    p2_x = np.random.rand(n_samples) * 5.0
    p2_y = np.random.rand(n_samples) * 5.0
    or2 = np.random.rand(n_samples) * np.pi
    f2 = np.random.rand(n_samples) * 5.0
    ph2 = np.random.rand(n_samples) * np.pi
    
    # 1. Baseline (The Original Vectorized Method)
    # Important: The original method is actually quite good because it uses numpy vectorization.
    # However, it creates many temporary intermediate arrays.
    
    # Note: We must check if user ALREADY replaced the method in source code.
    # If replaced, we can't easily run the "slow" version unless we monkey-patch or copy code.
    # We will assume 'V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized'
    # points to whatever is currently in the file.
    
    print("\nRunning 'Current Implementation' (might be JIT if you already patched it)...")
    t0 = time.time()
    res_orig = V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized(
        K1, w1_x, w1_y, p1_x, p1_y, or1, f1, or1, ph1,
        K2, w2_x, w2_y, p2_x, p2_y, or2, f2, or2, ph2
    )
    t_current = time.time() - t0
    print(f"  Current Implementation Time: {t_current:.4f} s")

    # 2. Explicit Fast JIT (To compare, in case the above is still the slow one)
    if JIT_AVAILABLE:
        print("\nRunning 'fast_integral_vectorized' (Direct JIT call)...")
        # Warmup
        fast_integral_vectorized(
            K1[:10], w1_x[:10], w1_y[:10], p1_x[:10], p1_y[:10], or1[:10], f1[:10], or1[:10], ph1[:10],
            K2[:10], w2_x[:10], w2_y[:10], p2_x[:10], p2_y[:10], or2[:10], f2[:10], or2[:10], ph2[:10]
        )
        
        t0 = time.time()
        res_fast = fast_integral_vectorized(
            K1, w1_x, w1_y, p1_x, p1_y, or1, f1, or1, ph1,
            K2, w2_x, w2_y, p2_x, p2_y, or2, f2, or2, ph2
        )
        t_fast = time.time() - t0
        print(f"  Numba JIT Time             : {t_fast:.4f} s")
        
        if t_current > t_fast * 1.2:
             print(f"  Speedup                    : {t_current / t_fast:.1f}x (vs Current)")
             print("  >> The current implementation seems to be the SLOW version.")
        elif t_current < t_fast * 1.2 and t_current > t_fast * 0.8:
             print("  >> The current implementation matches JIT speed.")
             print("     (You have likely already patched the source code!)")
        else:
             print(f"  Speedup                    : {t_current / t_fast:.1f}x")


if __name__ == "__main__":
    print("=========================================")
    print("   MOZAIK JIT OPTIMIZATION BENCHMARK     ")
    print("=========================================")
    
    run_geometric_benchmark(n_calls=200000)
    run_rf_update_benchmark(spatial_dim=30, duration=50, n_steps=5000)
    run_connectivity_integral_benchmark(n_samples=50000)
    
    print("\nDone.")