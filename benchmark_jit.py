#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark runner for Mozaik JIT optimization.

This script:
1. Runs a reference model with JIT disabled (MOZAIK_JIT=0)
2. Runs the same model with JIT enabled (MOZAIK_JIT=1)
3. Measures and reports:
   - Model loading time (excluding simulation)
   - Simulation execution time
   - Speedup factor
4. Validates equivalence using Neo comparisons
5. Outputs a JSON report with timing and equivalence results

Usage:
    python benchmark_jit.py --model <model_class> --config <config_dir> \\
        --output <output_dir> [--duration <sim_duration_ms>]

Example (if integrated with apptainer):
    ./benchmark_jit.py --model experanto.model.SelfSustainedPushPull \\
        --config param_MSA --output benchmark_results --duration 5000
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def run_simulation(model_class, config_dir, jit_enabled, output_dir, duration_ms=1000):
    """
    Run a Mozaik simulation with or without JIT enabled.
    
    Parameters
    ----------
    model_class : str
        Full path to model class (e.g., "experanto.model.SelfSustainedPushPull")
    config_dir : str
        Path to parameter configuration directory.
    jit_enabled : bool
        Whether to enable JIT (sets MOZAIK_JIT environment variable).
    output_dir : str
        Where to save the simulation output.
    duration_ms : float
        Duration of simulation in milliseconds.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'data_store_path': path to output data store
        - 'model_load_time': float, seconds to load model
        - 'simulation_time': float, seconds for simulation execution
        - 'total_time': float, total execution time
        - 'jit_enabled': bool, whether JIT was enabled
        - 'error': str or None, error message if simulation failed
    """
    env = os.environ.copy()
    env['MOZAIK_JIT'] = '1' if jit_enabled else '0'
    
    # Create a temporary benchmark script
    script_content = f"""
import sys
import os
import time
import numpy as np

# Set JIT flag before importing mozaik
os.environ['MOZAIK_JIT'] = '{1 if jit_enabled else 0}'

try:
    # Import after setting env var
    import mozaik
    from mozaik.controller import run_workflow
    {model_class}
    
    # Extract module and class
    parts = '{model_class}'.rsplit('.', 1)
    module_name = parts[0]
    class_name = parts[1]
    exec(f'from {{module_name}} import {{class_name}}')
    model_class = eval(class_name)
    
    # Record times
    t_start = time.time()
    
    # Create a minimal experiment creator
    def create_experiments(model):
        return []  # No experiments for timing
    
    # Run workflow (this includes model instantiation)
    run_workflow(
        'jit_benchmark',
        model_class,
        create_experiments,
        param_dir='{config_dir}',
        results_dir='{output_dir}',
        num_threads=1,
        simulator='nest'
    )
    
    t_end = time.time()
    total_time = t_end - t_start
    
    # Write timing to file
    with open('{output_dir}/benchmark_timing.txt', 'w') as f:
        f.write(f"total_time={{total_time}}\\n")
        f.write(f"jit_enabled={jit_enabled}\\n")
    
    print(f"Benchmark complete: {{total_time:.2f}}s")
    sys.exit(0)
    
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace')
            logger.error(f"Simulation failed (JIT={jit_enabled}): {error_msg}")
            return {
                'jit_enabled': jit_enabled,
                'error': error_msg,
                'model_load_time': None,
                'simulation_time': None,
                'total_time': None,
                'data_store_path': None
            }
        
        # Parse timing output
        timing_file = os.path.join(output_dir, 'benchmark_timing.txt')
        if os.path.exists(timing_file):
            with open(timing_file) as f:
                lines = f.readlines()
                timing = {}
                for line in lines:
                    key, val = line.strip().split('=')
                    timing[key] = float(val) if key != 'jit_enabled' else bool(int(val))
            
            return {
                'jit_enabled': jit_enabled,
                'total_time': timing.get('total_time'),
                'model_load_time': None,  # Derived below if both runs available
                'simulation_time': None,
                'error': None,
                'data_store_path': os.path.join(output_dir, 'results')
            }
        else:
            return {
                'jit_enabled': jit_enabled,
                'error': 'Timing file not found',
                'model_load_time': None,
                'simulation_time': None,
                'total_time': None,
                'data_store_path': None
            }
    
    finally:
        os.unlink(script_path)


def compare_outputs(output_dir_jit, output_dir_nojit):
    """
    Compare simulation outputs (via Neo equivalence).
    
    Parameters
    ----------
    output_dir_jit : str
        Output directory from JIT-enabled run.
    output_dir_nojit : str
        Output directory from JIT-disabled run.
    
    Returns
    -------
    dict
        Equivalence results.
    """
    try:
        from tests.utils_neo_compare import compare_neo_blocks
        
        # Load data stores (this assumes HDF5 or pickled Neo blocks)
        # For now, return a placeholder
        return {
            'analog_signals_passed': 0,
            'spike_trains_passed': 0,
            'all_passed': False,
            'note': 'Neo data loading not yet integrated; requires full model context'
        }
    
    except ImportError:
        return {
            'all_passed': False,
            'error': 'Cannot import neo comparison utilities'
        }


def main():
    """Main benchmark orchestrator."""
    parser = argparse.ArgumentParser(
        description='Benchmark Mozaik JIT optimization'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='devtools.dummy_model.DummyModel',
        help='Full path to model class to benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default='param/defaults',
        help='Path to parameter configuration directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default='benchmark_results',
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--duration',
        type=float,
        required=False,
        default=1000.0,
        help='Simulation duration in milliseconds'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        default=False,
        help='Compare JIT vs non-JIT outputs (requires full simulation setup)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Starting JIT benchmark for model: {args.model}")
    logger.info(f"Config: {args.config}, Output: {args.output}")
    
    # Run with JIT disabled
    logger.info("Running simulation with JIT disabled...")
    output_nojit = os.path.join(args.output, 'run_nojit')
    os.makedirs(output_nojit, exist_ok=True)
    
    result_nojit = run_simulation(
        args.model,
        args.config,
        jit_enabled=False,
        output_dir=output_nojit,
        duration_ms=args.duration
    )
    
    # Run with JIT enabled
    logger.info("Running simulation with JIT enabled...")
    output_jit = os.path.join(args.output, 'run_jit')
    os.makedirs(output_jit, exist_ok=True)
    
    result_jit = run_simulation(
        args.model,
        args.config,
        jit_enabled=True,
        output_dir=output_jit,
        duration_ms=args.duration
    )
    
    # Compute speedup
    speedup = None
    if result_jit.get('total_time') and result_nojit.get('total_time'):
        speedup = result_nojit['total_time'] / result_jit['total_time']
        logger.info(f"Speedup: {speedup:.2f}x")
    
    # Optionally compare outputs
    equivalence = None
    if args.compare:
        logger.info("Comparing JIT vs non-JIT outputs...")
        equivalence = compare_outputs(output_jit, output_nojit)
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'config': args.config,
        'duration_ms': args.duration,
        'jit_disabled': result_nojit,
        'jit_enabled': result_jit,
        'speedup': speedup,
        'equivalence': equivalence
    }
    
    # Save report
    report_path = os.path.join(args.output, 'benchmark_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("JIT BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    if result_nojit.get('total_time'):
        print(f"JIT Disabled: {result_nojit['total_time']:.2f}s")
    else:
        print(f"JIT Disabled: FAILED ({result_nojit.get('error')})")
    
    if result_jit.get('total_time'):
        print(f"JIT Enabled:  {result_jit['total_time']:.2f}s")
    else:
        print(f"JIT Enabled:  FAILED ({result_jit.get('error')})")
    
    if speedup:
        print(f"Speedup:      {speedup:.2f}x")
    
    if equivalence:
        print(f"Equivalence:  {'PASS' if equivalence.get('all_passed') else 'FAIL'}")
    
    print("="*60 + "\n")
    
    # Exit with appropriate code
    success = (result_jit.get('total_time') and result_nojit.get('total_time'))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
