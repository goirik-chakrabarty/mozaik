import time
import numpy as np
import mozaik.connectors.vision as vision
from mozaik.connectors.vision import V1CorrelationBasedConnectivity

def benchmark_connectivity_integrals(n_samples=1000):
    """
    Benchmarks the heavy linear algebra used in V1 connectivity.
    Target for JIT: V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized
    """
    print(f"\n--- Benchmarking Connectivity Integrals ({n_samples} calls) ---")
    
    # Generate random dummy data representing Gabor parameters
    # Shapes mimic the vectorized calls in the actual connector
    K1 = np.random.rand(n_samples)
    w1_x = np.random.rand(n_samples) * 0.5
    w1_y = np.random.rand(n_samples) * 0.5
    p1_x = np.random.rand(n_samples) * 5.0
    p1_y = np.random.rand(n_samples) * 5.0
    or1 = np.random.rand(n_samples) * np.pi
    f1 = np.random.rand(n_samples) * 5.0
    ph1 = np.random.rand(n_samples) * np.pi
    
    # Target params (same shape)
    K2 = np.random.rand(n_samples)
    w2_x = np.random.rand(n_samples) * 0.5
    w2_y = np.random.rand(n_samples) * 0.5
    p2_x = np.random.rand(n_samples) * 5.0
    p2_y = np.random.rand(n_samples) * 5.0
    or2 = np.random.rand(n_samples) * np.pi
    f2 = np.random.rand(n_samples) * 5.0
    ph2 = np.random.rand(n_samples) * np.pi
    
    # Use the static method directly
    start_time = time.time()
    
    # We loop to simulate the cumulative load of millions of connections
    # Note: The actual function handles vectors, but is often called in chunks.
    # We pass the large vectors once to see vectorized performance.
    result = V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized(
        K1, w1_x, w1_y, p1_x, p1_y, or1, f1, or1, ph1,
        K2, w2_x, w2_y, p2_x, p2_y, or2, f2, or2, ph2
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Speed: {n_samples / duration:.2f} integrals/sec")
    return duration

def benchmark_geometric_kernels(n_calls=100000):
    """
    Benchmarks the 'gabor' and 'gauss' function calls.
    Target for JIT: mozaik.connectors.vision.gabor / gauss
    """
    print(f"\n--- Benchmarking Geometric Kernels ({n_calls} calls) ---")
    
    x1, y1 = 0.5, 0.5
    x2 = np.random.rand(n_calls)
    y2 = np.random.rand(n_calls)
    orientation = 0.45
    freq = 5.0
    phase = 0.0
    size = 0.2
    aspect_ratio = 1.0
    
    start_time = time.time()
    
    # Simulate the loop found in GaborArborization.evaluate
    # We manually vectorise the call here to stress the math function
    # In the real code, this is often a loop or a list comprehension
    
    # 1. Benchmark Gabor
    res_gabor = vision.gabor(x1, y1, x2, y2, orientation, freq, phase, size, aspect_ratio)
    
    # 2. Benchmark Gauss
    res_gauss = vision.gauss(x1, y1, x2, y2, orientation, size, aspect_ratio)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Speed: {n_calls / duration:.2f} evals/sec")
    return duration

def benchmark_rf_update(spatial_dim=50, duration=100, n_steps=500):
    """
    Benchmarks the receptive field 'view' update logic.
    Target for JIT: CellWithReceptiveField.view logic
    """
    print(f"\n--- Benchmarking RF View Update ({n_steps} steps) ---")
    
    # Mock data
    # Kernel: (Spatial_Pixels, Temporal_Duration)
    kernel_contrast = np.random.rand(spatial_dim**2, duration)
    # View Array: (Spatial_Pixels_X, Spatial_Pixels_Y)
    view_array = np.random.rand(spatial_dim, spatial_dim)
    background_lum = 0.5
    
    # Response buffer
    contrast_response = np.zeros(n_steps + duration)
    
    start_time = time.time()
    
    for i in range(n_steps):
        # This is the exact bottleneck line from spatiotemporalfilter.py
        # reshaped manually here to match the internal logic
        flat_view = view_array.reshape(-1)
        scaled_view = flat_view / background_lum
        
        # The Dot Product Bottleneck
        contrast_tc = np.dot(kernel_contrast.T, scaled_view)
        
        # The Accumulation Bottleneck
        contrast_response[i : i+duration] += contrast_tc
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    return duration

if __name__ == "__main__":
    print("=== Mozaik JIT Benchmark ===")
    print("Running benchmarks on current implementation...")
    
    try:
        t1 = benchmark_connectivity_integrals(n_samples=200000)
        t2 = benchmark_geometric_kernels(n_calls=200000)
        t3 = benchmark_rf_update(spatial_dim=30, duration=50, n_steps=5000)
        
        print("\nTotal Benchmark Time: {:.4f} s".format(t1 + t2 + t3))
        print("Note: If you have enabled JIT, the first run includes compilation time.")
        print("      Run this script twice to see the cached (warm) performance.")
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        print("Ensure you are running this from the root folder where 'mozaik' is importable.")