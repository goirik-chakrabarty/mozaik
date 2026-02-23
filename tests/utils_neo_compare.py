# -*- coding: utf-8 -*-
"""
Utilities for comparing Neo data structures to validate JIT equivalence.

Provides tools to extract AnalogSignal and SpikeTrain data from Neo Block objects
and compute numeric distances (L_inf, RMSE) for validation that JIT and non-JIT
versions produce output within acceptable floating-point tolerances.
"""

import numpy as np
from neo import Block, SpikeTrain, AnalogSignal
from quantities import Quantity


def extract_analog_signals(block, unit_id=None):
    """
    Extract all AnalogSignal objects from a Neo Block.
    
    Parameters
    ----------
    block : neo.Block
        The Neo Block containing segments with AnalogSignal data.
    unit_id : str, optional
        Optional unit ID to filter signals.
    
    Returns
    -------
    dict
        Dictionary mapping (segment_idx, signal_name) -> AnalogSignal
    """
    signals = {}
    for seg_idx, segment in enumerate(block.segments):
        for signal in segment.analogsignals:
            key = (seg_idx, signal.name if hasattr(signal, 'name') else f"signal_{seg_idx}")
            signals[key] = signal
    return signals


def extract_spike_trains(block, unit_id=None):
    """
    Extract all SpikeTrain objects from a Neo Block.
    
    Parameters
    ----------
    block : neo.Block
        The Neo Block containing segments with SpikeTrain data.
    unit_id : str, optional
        Optional unit ID to filter spike trains.
    
    Returns
    -------
    dict
        Dictionary mapping (segment_idx, unit_name) -> SpikeTrain
    """
    spike_trains = {}
    for seg_idx, segment in enumerate(block.segments):
        for unit_idx, unit in enumerate(segment.units):
            for st in unit.spiketrains:
                key = (seg_idx, unit.name if hasattr(unit, 'name') else f"unit_{unit_idx}")
                spike_trains[key] = st
    return spike_trains


def compute_analog_signal_distance(signal1, signal2, metric='both'):
    """
    Compute distance between two AnalogSignal objects.
    
    Parameters
    ----------
    signal1 : neo.AnalogSignal
        First signal (magnitude array).
    signal2 : neo.AnalogSignal
        Second signal (magnitude array).
    metric : str
        'linf' for L_infinity norm, 'rmse' for root mean square error, 'both' for both.
    
    Returns
    -------
    dict or float
        If metric=='linf' or 'rmse': single float value.
        If metric=='both': dict with 'linf' and 'rmse' keys.
    
    Raises
    ------
    ValueError
        If signals have different shapes or units don't match.
    """
    # Ensure both signals have same units and shape
    s1_mag = signal1.magnitude
    s2_mag = signal2.rescale(signal1.units).magnitude
    
    if s1_mag.shape != s2_mag.shape:
        raise ValueError(f"Signal shapes don't match: {s1_mag.shape} vs {s2_mag.shape}")
    
    diff = s1_mag - s2_mag
    
    if metric == 'linf':
        return float(np.max(np.abs(diff)))
    elif metric == 'rmse':
        return float(np.sqrt(np.mean(diff ** 2)))
    elif metric == 'both':
        return {
            'linf': float(np.max(np.abs(diff))),
            'rmse': float(np.sqrt(np.mean(diff ** 2)))
        }
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compare_spike_trains(st1, st2, tolerance=None):
    """
    Compare two SpikeTrain objects for equivalence.
    
    Parameters
    ----------
    st1 : neo.SpikeTrain
        First spike train.
    st2 : neo.SpikeTrain
        Second spike train.
    tolerance : Quantity, optional
        Maximum allowed temporal shift between corresponding spikes.
        If None, defaults to the integration timestep.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'total_spike_count_match': bool, True if spike counts are identical
        - 'count1': int, spike count in st1
        - 'count2': int, spike count in st2
        - 'max_temporal_shift': Quantity, maximum difference in spike times
        - 'shifts': array, all individual spike time differences
        - 'passed': bool, True if counts match and shifts within tolerance
    """
    if st1.units != st2.units:
        st2_rescaled = st2.rescale(st1.units).magnitude
        st1_mag = st1.magnitude
    else:
        st1_mag = st1.magnitude
        st2_rescaled = st2.magnitude
    
    count1 = len(st1_mag)
    count2 = len(st2_rescaled)
    
    result = {
        'total_spike_count_match': (count1 == count2),
        'count1': count1,
        'count2': count2,
    }
    
    if count1 == 0 and count2 == 0:
        result['max_temporal_shift'] = 0.0 * st1.units
        result['shifts'] = np.array([]) * st1.units
        result['passed'] = True
        return result
    
    if count1 != count2:
        result['max_temporal_shift'] = np.inf * st1.units
        result['shifts'] = np.array([]) * st1.units
        result['passed'] = False
        return result
    
    shifts = np.abs(st1_mag - st2_rescaled) * st1.units
    result['max_temporal_shift'] = np.max(shifts)
    result['shifts'] = shifts
    
    if tolerance is None:
        # Default tolerance: Integration timestep (1 ms is common, or derive from sampling rate)
        tolerance = 1.0  # ms by default
        if hasattr(st1, 'sampling_rate') and st1.sampling_rate is not None:
            tolerance = (1.0 / st1.sampling_rate).rescale('ms')
    
    result['passed'] = (count1 == count2) and (np.max(shifts) <= tolerance)
    
    return result


def compare_neo_blocks(block1, block2, analog_tolerances=None, spike_tolerance=None):
    """
    Comprehensive comparison of two Neo Block objects.
    
    Parameters
    ----------
    block1 : neo.Block
        First block (e.g., JIT-enabled output).
    block2 : neo.Block
        Second block (e.g., JIT-disabled output).
    analog_tolerances : dict, optional
        Dictionary with keys 'linf' (default 1e-6) and 'rmse' (default 1e-7) for tolerances.
    spike_tolerance : Quantity, optional
        Tolerance for spike timing differences.
    
    Returns
    -------
    dict
        Comprehensive results including:
        - 'analog_signal_results': list of dicts, one per signal
        - 'spike_train_results': list of dicts, one per spike train
        - 'all_passed': bool, True if all comparisons passed
        - 'summary': str, human-readable summary
    """
    if analog_tolerances is None:
        analog_tolerances = {'linf': 1e-6, 'rmse': 1e-7}
    
    results = {
        'analog_signal_results': [],
        'spike_train_results': [],
        'all_passed': True,
        'summary': ''
    }
    
    # Compare AnalogSignals
    signals1 = extract_analog_signals(block1)
    signals2 = extract_analog_signals(block2)
    
    for key in signals1.keys():
        if key not in signals2:
            results['analog_signal_results'].append({
                'key': key,
                'passed': False,
                'reason': 'Signal missing in block2'
            })
            results['all_passed'] = False
            continue
        
        distances = compute_analog_signal_distance(signals1[key], signals2[key], metric='both')
        passed = (distances['linf'] <= analog_tolerances['linf'] and 
                  distances['rmse'] <= analog_tolerances['rmse'])
        
        results['analog_signal_results'].append({
            'key': key,
            'linf': distances['linf'],
            'rmse': distances['rmse'],
            'linf_tol': analog_tolerances['linf'],
            'rmse_tol': analog_tolerances['rmse'],
            'passed': passed
        })
        
        if not passed:
            results['all_passed'] = False
    
    # Compare SpikeTrain
    st1_dict = extract_spike_trains(block1)
    st2_dict = extract_spike_trains(block2)
    
    for key in st1_dict.keys():
        if key not in st2_dict:
            results['spike_train_results'].append({
                'key': key,
                'passed': False,
                'reason': 'SpikeTrain missing in block2'
            })
            results['all_passed'] = False
            continue
        
        st_comparison = compare_spike_trains(st1_dict[key], st2_dict[key], tolerance=spike_tolerance)
        results['spike_train_results'].append({
            'key': key,
            **st_comparison
        })
        
        if not st_comparison['passed']:
            results['all_passed'] = False
    
    # Build summary
    analog_passed = sum(1 for r in results['analog_signal_results'] if r.get('passed', False))
    analog_total = len(results['analog_signal_results'])
    spike_passed = sum(1 for r in results['spike_train_results'] if r.get('passed', False))
    spike_total = len(results['spike_train_results'])
    
    results['summary'] = (
        f"Analog signals: {analog_passed}/{analog_total} passed; "
        f"Spike trains: {spike_passed}/{spike_total} passed; "
        f"Overall: {'PASS' if results['all_passed'] else 'FAIL'}"
    )
    
    return results
