from mozaik.storage.queries import param_filter_query
from mozaik.tools.mozaik_parametrized import MozaikParametrized
from mozaik.tools.distribution_parametrization import PyNNDistribution
import os
import numpy as np
from collections import OrderedDict
import h5py
import pickle
import json
from quantities import Quantity
import copy
import h5py
import numpy as np
import contextlib
import logging
import neo
import gc 
import multiprocessing

def export_from_datastore_to_hdf5_with_multiprocessing(data_store, st_name, data_type, start_time=None, stop_time=None, time_windows_size=None, path_to_save_hdf5=None):
    """
    Export data from a Mozaik datastore to a HDF5 file with a standardized structure.

    Parameters:
        data_store (DataStore): The Mozaik datastore containing simulation results
        st_name (str): The name of the stimulus to be exported
        data_type (str): The type of data to export. Options:
                        - 'spike_counts': Number of spikes in each time window
                        - 'mean_rates': Average firing rates
                        - 'spiketrains': Raw spike times
        start_time (float, optional): Start time (ms) for data extraction
        stop_time (float, optional): End time (ms) for data extraction
        time_windows_size (float, optional): Size of the time windows for data extraction
        path_to_save_hdf5 (str, optional): Path to save the HDF5 file. If None, the file is saved in the base folder of the data_store.

    Notes:
        Creates an HDF5 file with the following structure:
        - Root attributes: default_parameters, sim_info, sheets, data_type, st_name, recorders
        - Model groups: Contains parameter sets and stimulus data
        - Stimulus groups: Contains:
            - Constant/varying parameter information
            - Neural response data for each sheet
            - Stimulus data and indices
            - Data cut timing information

    """
    print("Starting export_from_datastore_to_hdf5")

    def serialize_parameters(params):
        serialized = {}
        for key, value in params.items():
            if isinstance(value, dict):
                serialized[key] = serialize_parameters(value)
            elif isinstance(value, PyNNDistribution):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized
    
    def create_hdf5_structure(hf, sim_info, default_parameters, modified_parameters, recorders, experimental_protocols, st_name, varying_stim_params, constant_stim_params, data_type, start_time, stop_time):
        print("Creating HDF5 structure")
        # Add default parameters and info as metadata to the group
        hf.attrs['default_parameters'] = str(serialize_parameters(default_parameters))
        hf.attrs['sim_info'] = str(sim_info)
        hf.attrs['data_type'] = data_type
        hf.attrs['st_name'] = st_name
        hf.attrs['recorders'] = str(recorders)
        hf.attrs['experimental_protocols'] = str(experimental_protocols)

        # Create a subgroup based on modified parameters
        if modified_parameters:
            model_subgroup_name = str(modified_parameters)
        else:
            model_subgroup_name = "default"
        model_subgroup = hf.create_group(model_subgroup_name)

        # Merge modified parameters into default parameters
        merged_parameters = default_parameters.copy()
        merged_parameters.update(modified_parameters)

        # Add merged parameters as metadata to the model_subgroup
        model_subgroup.attrs['parameters'] = str(serialize_parameters(merged_parameters))
        logging.info(f"Model subgroup '{model_subgroup_name}' created with merged parameters as metadata.")

        # Create a stimuli subgroup
        stimuli_subgroup = model_subgroup.create_group(st_name)
        logging.info(f"Datasets subgroup created under 'stimuli' in '{model_subgroup_name}'.")

        # Add varying parameters as metadata to the stimuli subgroup
        stimuli_subgroup.attrs['varying_parameters'] = list(varying_stim_params.keys())
        for param_name, param_values in varying_stim_params.items():
            stimuli_subgroup.attrs[f'{param_name}'] = param_values
        stimuli_subgroup.attrs['data_dimensions'] = [len(varying_stim_params[param]) for param in varying_stim_params.keys()]

        # Add constant parameters as metadata to the stimuli subgroup
        stimuli_subgroup.attrs['constant_parameters'] = list(constant_stim_params.keys())
        for param_name, param_values in constant_stim_params.items():
            stimuli_subgroup.attrs[f'{param_name}'] = str(param_values) if param_values is not None else "None"

        # Add data related metadata to the stimuli subgroup
        stimuli_subgroup.attrs['data_type'] = data_type
        stimuli_subgroup.attrs['data_start_time'] = str(start_time) if start_time is not None else "None"
        stimuli_subgroup.attrs['data_stop_time'] = str(stop_time) if stop_time is not None else "None"

        print("Finished creating HDF5 structure")
        return stimuli_subgroup

    def get_segments_and_stimuli_and_constant_and_varying_parameters(data_store, sheet_name, st_name):
        print(f"Getting segments and parameters for sheet {sheet_name}")
        # Get segments and stimuli
        dsv = param_filter_query(data_store, st_name=st_name, sheet_name=sheet_name)
        segs = dsv.get_segments()
        stims = [MozaikParametrized.idd(seg.annotations['stimulus']) for seg in segs]
        segs_pre =  dsv.get_segments(null=True)
        segs_post = [*segs_pre[1:], param_filter_query(data_store, st_name='InternalStimulus', sheet_name=sheet_name).get_segments()[-1]]
        segs = list(zip(segs_pre, segs, segs_post))

        print("Getting varying parameters")
        # Get varying parameters
        constant_stim_params, varying_stim_params = classify_stimulus_parameters_into_constant_and_varying(stims)  # alternative: params = OrderedDict((param, sorted(list(parameter_value_list(stims, param)))) for param in varying_parameters(stims))

        # Assert all possible combinations of varying stimulus parameters are present in the stimuli in the list stims and that there are no duplicates
        assert len(stims) == np.prod([len(varying_stim_params[param]) for param in varying_stim_params.keys()]), "Number of stimuli does not match the product of the number of varying parameter values"
        assert len(set([str(stim) for stim in stims])) == len(stims), "There are duplicate stimuli"
        print("Finished getting segments and parameters")
        return segs, stims, constant_stim_params, varying_stim_params


    def extract_sheet_data_and_save_to_h5py(stims, segs, varying_stim_params, data_type, stimuli_subgroup, sheet_name, start_time, stop_time, time_windows_size):
        print(f"Starting data extraction for sheet {sheet_name}")
        # Get data to export
        logging.info(f"Extracting {data_type} data from {len(segs)} segments in sheet {sheet_name}")

        # define start_time and stop_time if not provided
        if start_time is None:
            start_time = 0
        if stop_time is None:
            stop_time = stims[0].duration

        # define time_windows
        if time_windows_size == None: 
            time_windows = [(start_time, stop_time)]
            time_windows_size = stop_time - start_time
        else:
            assert float(stop_time-start_time) % (time_windows_size) == 0, "time_windows_size must be a multiple of the time_stop - time_start"
            time_windows = list(zip(np.arange(start_time, stop_time, time_windows_size), np.arange(start_time+time_windows_size, stop_time+time_windows_size, time_windows_size)))

        segs_to_concatenate_identifiers = []
        for seg_pre, seg, seg_post in segs:
            segs_to_concatenate_identifiers.append((seg_pre.datastore_path + '/' + seg_pre.identifier + ".pickle", seg.datastore_path + '/' + seg.identifier + ".pickle", seg_post.datastore_path + '/' + seg_post.identifier + ".pickle"))

        print("Processing segments in parallel")
        # Prepare arguments for parallel processing
        process_args = [(segs_identifiers, data_type, time_windows, start_time, stop_time, time_windows_size) 
                    for segs_identifiers in segs_to_concatenate_identifiers]
    

        # Use multiprocessing to process segments in parallel
        with multiprocessing.Pool() as pool:
            data = pool.map(process_segment_from_identifiers, process_args)
    

        print("Converting data to array")
        if data_type == 'spiketrains':
            data = np.array(data, dtype=object)
        else:
            data = np.stack(data)

        print("Reordering data")
        # Reorder stimuli and data in tensors whose number of dimensions corresponds to the number of varying parameters
        _, data_sorted = reorder_lists(stims, data, varying_stim_params.keys())

        # fix 2
        del data 
        gc.collect()

        print("Reshaping data tensor")
        params_dims = [len(varying_stim_params[param]) for param in varying_stim_params.keys()]
        # reshape data to match the dimensions of varying parameters
        data_tensor = np.reshape(np.array(data_sorted).flatten(), [*params_dims, *data_sorted[0].shape])

        # fix 3
        del data_sorted
        gc.collect()

        print("Saving data to HDF5")
        # Add dataset to the stimuli subgroup
        sheet_name_cleaned = sheet_name.replace('/', '')
        if data_type =='spiketrains':
            dset = stimuli_subgroup.create_dataset(sheet_name_cleaned, shape=data_tensor.shape, dtype=h5py.special_dtype(vlen=np.dtype('float')))
            dset[:] = data_tensor
        else:
            stimuli_subgroup.create_dataset(sheet_name_cleaned, data=data_tensor)
        print(f"Finished processing sheet {sheet_name}")

    def add_stimuli_dataset(stimuli_subgroup, stims, varying_stim_params, ds):
        print("Adding stimuli dataset")
        logging.info(f"Adding stimuli dataset to {stimuli_subgroup.name}")
        # Reorder stimuli and reshape to match the dimensions of varying parameters
        reordered_stims, _ = reorder_lists(stims, [str(s) for s in stims], varying_stim_params.keys()) 
        reordered_stims = np.array(reordered_stims).reshape([len(varying_stim_params[param]) for param in varying_stim_params.keys()])

        # Identify which dimension corresponds to trial
        trial_dim = None
        for i, param in enumerate(varying_stim_params.keys()):
            if param == 'trial':
                trial_dim = i
                stimuli_subgroup.attrs['trial_dim'] = trial_dim
                break

        # Drop the trial dimension by selecting the first element along it
        if trial_dim != None:
            reordered_stims = np.take(reordered_stims, 0, axis=trial_dim)
            if type(reordered_stims) != np.ndarray:
                reordered_stims = np.array([reordered_stims])
                
        # create index
        reordered_stims_flat = reordered_stims.flatten()
        reordered_stims_idx = np.arange(len(reordered_stims_flat)).reshape(reordered_stims.shape)
        
        # reinsert trial dimension
        if trial_dim != None:
            reordered_stims_idx = np.expand_dims(reordered_stims_idx, axis=trial_dim).repeat(len(varying_stim_params['trial']), axis=trial_dim)

        print("Getting sensory stimulus")
        sensory_stim = np.array(ds.get_sensory_stimulus([str(s) for s in reordered_stims_flat])).squeeze()
        stimuli_subgroup.create_dataset('stimuli', data=sensory_stim)   
        stimuli_subgroup.create_dataset('stimuli_idx', data=reordered_stims_idx)

        # Save the stimulus_name dataset
        stimulus_names = np.array([str(stim) for stim in stims], dtype='S')
        stimuli_subgroup.create_dataset('stimulus_name', data=stimulus_names, dtype=h5py.string_dtype(encoding='utf-8'))
        print("Finished adding stimuli dataset")

    ############################################################################################
    ## Create an HDF5 file (main function)
    ############################################################################################
    print("Setting up HDF5 file path")
    base_folder = data_store.parameters['root_directory']
    if path_to_save_hdf5 is None:
        path_to_save_hdf5 = os.path.join(base_folder, 'exported_data.h5')
    if path_to_save_hdf5.endswith('.h5'):

        if '/' not in path_to_save_hdf5:
            path_to_save_hdf5 = os.path.join('.', path_to_save_hdf5)
    assert path_to_save_hdf5.endswith('.h5'), "path_to_save_hdf5 must end with .h5"
    os.makedirs(os.path.dirname(path_to_save_hdf5), exist_ok=True)
    
    print(f"Creating HDF5 file at {path_to_save_hdf5}")
    with h5py.File(path_to_save_hdf5, 'w') as hf:
        # Get model info and parameters
        print("Getting model info and parameters")
        modified_parameters, default_parameters, info, recorders, experimental_protocols = get_model_info_and_parameters(base_folder, separate_modified_params=True)
        sheets =  data_store.sheets() 
        hf.attrs['sheets'] = [sheet.replace('/', '') for sheet in sheets]
        
        print(f"Processing {len(sheets)} sheets")
        # Iterate over all sheets, extract data and save to h5py
        for i, sheet_name in enumerate(sheets):
            print(f"\nProcessing sheet {i+1}/{len(sheets)}: {sheet_name}")
            # Get segments and stimuli and constant and varying parameters for the current sheet
            segs, stims, constant_stim_params, varying_stim_params = get_segments_and_stimuli_and_constant_and_varying_parameters(data_store=data_store, sheet_name=sheet_name, st_name=st_name)
            
            # Create HDF5 structure for the first sheet
            if i == 0:
                stimuli_subgroup = create_hdf5_structure(
                    hf, info, default_parameters, modified_parameters, recorders, experimental_protocols, st_name,
                    varying_stim_params, constant_stim_params, data_type, start_time, stop_time
                )

            if len(segs[0][1].get_spiketrains()) == 0: # maybe there is a better way to check if there are neurons recorded in the sheet
                logging.warning(f"No neurons recorded in sheet {sheet_name}")
                continue
            
            # # Extract data and save to h5py
            extract_sheet_data_and_save_to_h5py(stims, segs, varying_stim_params, data_type, stimuli_subgroup, sheet_name, start_time, stop_time, time_windows_size) 

        # Add stimuli dataset
        add_stimuli_dataset(stimuli_subgroup, stims, varying_stim_params, data_store)

    print("Finished creating HDF5 file")
    logging.info(f"HDF5 file created with default parameters, info, list of sheets as metadata, stimuli subgroup, and datasets subgroup.")

def count_spikes_in_multiple_windows(segment, time_windows):
    """
    Counts spikes in multiple time windows for all spiketrains in a neo.Segment.

    Parameters:
        segment (neo.Segment): The segment containing spiketrains.
        time_windows (list of tuple): List of time window tuples (start_time, end_time) in the same units as the spiketrain.

    Returns:
        np.ndarray: 2D array where each row corresponds to a spiketrain, and each column corresponds to a time window count.
    """
    # Initialize an array to store counts for all spiketrains and time windows
    num_spiketrains = len(segment.spiketrains)
    num_windows = len(time_windows)
    counts = np.zeros((num_spiketrains, num_windows), dtype=int)

    # Convert time windows to arrays for vectorized processing
    window_starts = np.array([w[0] for w in time_windows])
    window_ends = np.array([w[1] for w in time_windows])

    # Iterate over spiketrains
    for i, spiketrain in enumerate(segment.spiketrains):
        spike_times = spiketrain.times.magnitude  # Extract spike times as a NumPy array
        
        # Use broadcasting to check spikes within each time window
        for j, (start, end) in enumerate(zip(window_starts, window_ends)):
            counts[i, j] = np.sum((spike_times >= start) & (spike_times < end))
    return counts.squeeze()
    
def concatenate_segments_with_offsets(pre_seg, seg, post_seg):
    """
    Concatenates three segments (pre_seg, seg, and post_seg) into a single segment.
    Combines spiketrains into a single spiketrain for each set of spiketrains across pre_seg, seg, and post_seg.

    Parameters:
        pre_seg (neo.Segment): The segment whose spiketrains will be placed in negative times.
        seg (neo.Segment): The main segment to keep as-is.
        post_seg (neo.Segment): The segment whose spiketrains will be appended after `seg`.

    Returns:
        neo.Segment: A new segment containing a single concatenated spiketrain for each set.
    """
    combined_segment = neo.Segment(name="CombinedSegment")

    # Calculate the duration of the main segment
    t_start_pre = - pre_seg.t_stop
    t_start_post = seg.t_stop
    # Iterate over spiketrains in the main segment
    for i, spiketrain in enumerate(seg.spiketrains):
        # Collect spiketrain times from pre_seg, seg, and post_seg
        pre_times = pre_seg.spiketrains[i].times + t_start_pre
        main_times = spiketrain.times
        post_times = post_seg.spiketrains[i].times + t_start_post

        # Concatenate the spike times
        all_times = pre_times.rescale(main_times.units).tolist() + \
                    main_times.tolist() + \
                    post_times.rescale(main_times.units).tolist()

        # Create a new spiketrain with concatenated times
        new_spiketrain = neo.SpikeTrain(
            sorted(all_times),  # Ensure times are sorted
            t_start=pre_seg.spiketrains[i].t_start + t_start_pre,
            t_stop=post_seg.spiketrains[i].t_stop + t_start_post,
            units=main_times.units
        )
        combined_segment.spiketrains.append(new_spiketrain)
        new_spiketrain

    return combined_segment

def count_spikes_in_window(segment, start_time, end_time):
    """
    Counts the number of spikes within a specific time window for each spiketrain in a neo.Segment.

    Parameters:
        segment (neo.Segment): The segment containing spiketrains.
        start_time (float or Quantity): Start time of the window (in the same units as the spiketrain).
        end_time (float or Quantity): End time of the window (in the same units as the spiketrain).

    Returns:
        np.ndarray: Array of spike counts for each spiketrain.
    """
    # Initialize counts
    counts = []

    for spiketrain in segment.spiketrains:
        spike_times = spiketrain.times.magnitude 
        window_mask = (spike_times >= start_time.magnitude) & (spike_times < end_time.magnitude)
        counts.append(np.count_nonzero(window_mask))

    return np.array(counts)

def read_file(file_path):
    """
    Read data from a file, attempting different formats if the file extension is not recognized.

    Args:
        file_path (str): Path to the file.

    Returns:
        object: Data loaded from the file, or None if unsuccessful.
        
    Notes:
        Attempts to read the file in the following order:
        1. As a pickle file
        2. As a JSON file 
        3. As plain text
        Logs success/failure messages for debugging.
    """
    try:
        # First, try to read as pickle
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Data successfully loaded from {file_path} as pickle")
            return data
        except:
            pass

        # If pickle fails, try JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logging.info(f"Data successfully loaded from {file_path} as JSON")
            return data
        except:
            pass

        # If JSON fails, try plain text
        try:
            with open(file_path, 'r') as f:
                data = f.read()
            logging.info(f"Data successfully loaded from {file_path} as plain text")
            return data
        except:
            pass

        # If all attempts fail, raise an exception
        raise ValueError(f"Unable to read file: {file_path}")

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
    
    return None

def reorder_lists(object_list, list_to_order, ordering_parameters):
    """
    Reorders `object_list` (list of MozaikParametrized objects) based on the attributes 
    specified in `ordering_parameters`, and applies the same reordering to `list_to_order`.
    
    Parameters:
        object_list (list): List of MozaikParametrized objects to be sorted.
        list_to_order (list): List that needs to be reordered in the same order as `object_list`.
        ordering_parameters (list): List of strings representing the attributes of the objects in 
                                  `object_list` to sort by.
    
    Returns:
        tuple: A tuple containing:
            - reordered_object_list (list): The sorted list of MozaikParametrized objects
            - reordered_list_to_order (list): The list_to_order sorted in the same order as reordered_object_list
            
    Notes:
        The function maintains the pairwise association between elements in both lists
        while sorting based on multiple attributes of the MozaikParametrized objects.
    """
    def sort_by_multiple_attributes(attributes):
        """
        Returns a sorting key function that can sort by multiple attributes of a MozaikParametrized object.
        """
        def key_func(obj):
            return tuple(getattr(obj, attr) for attr in attributes)
        return key_func

    # Combine the object list and the list to order to maintain pairwise association
    combined = list(zip(object_list, list_to_order))

    # Sort the combined list based on the specified attributes of the object list
    combined_sorted = sorted(combined, key=lambda pair: sort_by_multiple_attributes(ordering_parameters)(pair[0]))

    # Unzip the sorted pairs back into the object list and the other list
    reordered_object_list, reordered_list_to_order = zip(*combined_sorted)

    # Convert to list before returning
    return list(reordered_object_list), list(reordered_list_to_order)

def get_model_info_and_parameters(base_folder, separate_modified_params=False):
    """
    Retrieves and processes model information and parameters from the given base folder.

    Parameters:
        base_folder (str): The path to the base folder containing model information and parameters.
        separate_modified_params (bool): If True, return modified parameters separately. Default is False.

    Returns:
        If separate_modified_params is False:
            tuple: (merged_parameters, sim_info, recorders, experimental_protocols)
                - merged_parameters: Dictionary containing all parameters with modified ones overriding defaults
                - sim_info: Dictionary containing simulation information
                - recorders: Dictionary containing recorder configurations
                - experimental_protocols: Dictionary containing experimental protocol settings
                
        If separate_modified_params is True:
            tuple: (modified_parameters, default_parameters, sim_info, recorders, experimental_protocols)
                - modified_parameters: Dictionary containing only parameters that differ from defaults
                - default_parameters: Dictionary containing original default parameters
                - sim_info: Dictionary containing simulation information
                - recorders: Dictionary containing recorder configurations
                - experimental_protocols: Dictionary containing experimental protocol settings
                
    Notes:
        - Removes 'experiments.' parameters (as they are experiment parameters) and 'results_dir' from modified parameters (as saved elsewhere)
        - Compares remaining modified parameters with defaults to identify true modifications
    """
    modified_parameters_path = os.path.join(base_folder, 'modified_parameters.json')
    default_parameters_path = os.path.join(base_folder, 'parameters.json')
    sim_info_path = os.path.join(base_folder, 'sim_info.json')
    recorders_path = os.path.join(base_folder, 'recorders.json')
    experimental_protocols_path = os.path.join(base_folder, 'experimental_protocols.json')

    modified_parameters = read_file(modified_parameters_path)
    default_parameters = read_file(default_parameters_path)
    sim_info = read_file(sim_info_path)
    recorders = read_file(recorders_path)
    experimental_protocols = read_file(experimental_protocols_path)

    # Remove 'experiments.' parameters and 'results_dir' from modified_parameters
    modified_parameters = {k: v for k, v in modified_parameters.items() if not k.startswith('experiments.') and k != 'results_dir'}

    # Compare remaining items in modified_parameters with default_parameters
    for key in list(modified_parameters.keys()):
        if key in default_parameters and modified_parameters[key] == default_parameters[key]:
            del modified_parameters[key]

    if separate_modified_params:
        return modified_parameters, default_parameters, sim_info, recorders, experimental_protocols
    else:
        # Merge modified parameters into default parameters
        merged_parameters = default_parameters.copy()
        merged_parameters.update(modified_parameters)
        return merged_parameters, sim_info, recorders, experimental_protocols

def classify_stimulus_parameters_into_constant_and_varying(stims):
    """
    Classify the parameters of a list of stimuli into constant and varying categories.

    Parameters:
        stims (list): A list of stimulus objects.

    Returns:
        tuple: A tuple containing:
            - constant_params (OrderedDict): Parameters that remain constant across all stimuli
            - varying_params (OrderedDict): Parameters that vary across stimuli, with their sorted unique values
            
    Notes:
        - Ensures 'trial' parameter is the first key in varying_params if it exists
        - Uses OrderedDict to maintain consistent parameter ordering
    """
    constant_params = OrderedDict()
    varying_params = OrderedDict()

    for param in stims[0].getParams().keys():
        values = [getattr(stim, param) for stim in stims]
        if all(value == values[0] for value in values):
            constant_params[param] = values[0]
        else:
            varying_params[param] = sorted(set(values))

    # Ensure 'trial' is the first key if it exists in varying_params
    if 'trial' in varying_params:
        varying_params = OrderedDict([('trial', varying_params['trial'])] + 
                                     [(k, v) for k, v in varying_params.items() if k != 'trial'])

    return constant_params, varying_params

def worker_unpickle(file_info):
            """
            Worker function to load pickled files and return a dictionary of index: object.
            Args:
                file_info (list of tuples): List of (index, file_path).
            Returns:
                dict: {index: unpickled_object}
            """
            result = {}
            for index, file_path in file_info:
                try:
                    with open(file_path, 'rb') as f:
                        result[index] = pickle.load(f).spiketrains  # Unpickle the object
                    logger.info("Loaded " + str(index))
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    result[index] = None  # Handle errors gracefully
            logger.info("Leaving process")
            return result

def pre_load(dsv):
        """
        Force all segments in the DSV to load from permanent memory.
        """
        import multiprocessing
        num_proc = 32
        chunk_size = (len(dsv.block.segments) + num_proc - 1) // num_proc
        file_paths = [(i,s.datastore_path + '/' + s.identifier + ".pickle") for i,s in enumerate(dsv.block.segments)]
        chunks = [file_paths[i:i + chunk_size] for i in range(0, len(dsv.block.segments), chunk_size)]
        logger.info("Spawning pool")
        # Use multiprocessing.Pool to handle parallel processing
        with multiprocessing.Pool(processes=num_proc) as pool:
            # Map chunks of file paths to the worker function
            results = pool.map(worker_unpickle, chunks)
        logger.info("Finished pool")
        # Combine all the dictionaries returned by the worker processes
        for partial_result in results:
            for k in partial_result.keys():
                dsv.block.segments[k]._spiketrains = partial_result[k]
                dsv.block.segments[k]._analogsignals = []
                dsv.block.segments[k].full = True

def process_segment_from_identifiers(args):
    """
    Worker function to process a single segment.
    """
    segs_to_concatenate_identifiers, data_type, time_windows, start_time, stop_time, time_windows_size = args
    
    print("Processing segment")
            
    concatenated_seg = concatenate_segments_with_offsets_from_identifiers(*segs_to_concatenate_identifiers)
    
    if data_type == 'spike_counts':
        result = count_spikes_in_multiple_windows(concatenated_seg, time_windows)
    elif data_type == 'mean_rates':
        result = count_spikes_in_multiple_windows(concatenated_seg, time_windows)/float(time_windows_size) * 1000
    elif data_type == 'spiketrains':
        result = [spiketrain.time_slice(start_time, stop_time).magnitude for spiketrain in concatenated_seg.spiketrains]
    else:
        raise ValueError("Invalid data type")

    print("Deleting concatenated segment")
    del concatenated_seg
    gc.collect()
    return result
    
def concatenate_segments_with_offsets_from_identifiers(pre_seg_identifier, seg_identifier, post_seg_identifier):
    """
    Concatenates three segments (pre_seg, seg, and post_seg) into a single segment.
    Combines spiketrains into a single spiketrain for each set of spiketrains across pre_seg, seg, and post_seg.

    Parameters:
        pre_seg (neo.Segment): The segment whose spiketrains will be placed in negative times.
        seg (neo.Segment): The main segment to keep as-is.
        post_seg (neo.Segment): The segment whose spiketrains will be appended after `seg`.

    Returns:
        neo.Segment: A new segment containing a single concatenated spiketrain for each set.
    """
    
    with open(pre_seg_identifier, 'rb') as f:
        pre_seg = pickle.load(f)
    with open(seg_identifier, 'rb') as f:
        seg = pickle.load(f)
    with open(post_seg_identifier, 'rb') as f:
        post_seg = pickle.load(f)
    # Calculate the duration of the main segment
    t_start_pre = - pre_seg.t_stop
    t_start_post = seg.t_stop
    # Iterate over spiketrains in the main segment
    combined_segment = neo.Segment(name="CombinedSegment")
    for i, spiketrain in enumerate(seg.spiketrains):
        # Collect spiketrain times from pre_seg, seg, and post_seg
        pre_times = pre_seg.spiketrains[i].times + t_start_pre
        main_times = spiketrain.times
        post_times = post_seg.spiketrains[i].times + t_start_post

        # Concatenate the spike times
        all_times = pre_times.rescale(main_times.units).tolist() + \
                    main_times.tolist() + \
                    post_times.rescale(main_times.units).tolist()

        # Create a new spiketrain with concatenated times
        new_spiketrain = neo.SpikeTrain(
            sorted(all_times),  # Ensure times are sorted
            t_start=pre_seg.spiketrains[i].t_start + t_start_pre,
            t_stop=post_seg.spiketrains[i].t_stop + t_start_post,
            units=main_times.units
        )
        combined_segment.spiketrains.append(new_spiketrain)
        new_spiketrain

    return combined_segment

def merge_hdf5_files(file_list, output_file):
    """
    Merge multiple HDF5 files created by Mozaik into a single HDF5 file.

    Parameters:
        file_list (list): List of paths to the input HDF5 files to be merged
        output_file (str): Path to the output merged HDF5 file

    Notes:
        Requirements for merging:
        - Files must have identical structure except for one varying parameter
        - All files must have the same:
            - Default parameters (except results_dir)
            - Simulation info (except run_date)
            - Sheets configuration
            - Data type
            - Stimulus name
            - Recorders configuration
            
        The merged file:
        - Combines data along the dimension of the varying parameter
        - Preserves all metadata and attributes
        - Maintains the original file structure
        - Includes information about the merge operation:
            - Merging dimension name and index
            - Input sizes from each file
            - Combined parameter values
            
        Handles both regular datasets and variable-length datasets (e.g., spiketrains)
    """
    with h5py.File(output_file, 'w') as f_merged:
        # Initialize variables to store common information
        default_parameters = None
        sim_info = None
        results_dirs = []
        run_dates = []
        experimental_protocols = []
        sheets = None
        data_type = None
        st_name = None
        

        # Open all input files
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(h5py.File(file, 'r')) for file in file_list]

            # Check if all files have the same attributes
            for idx, f in enumerate(files):
                current_default_params = eval(f.attrs['default_parameters'])
                current_sim_info = eval(f.attrs['sim_info'])
                current_sheets = f.attrs['sheets']
                current_data_type = f.attrs['data_type']
                current_st_name = f.attrs['st_name']
                current_recorders = f.attrs['recorders']

                results_dirs.append(current_default_params.pop('results_dir'))
                run_dates.append(current_sim_info.pop('run_date'))
                experimental_protocols.append(f.attrs['experimental_protocols'])

                if idx == 0:
                    default_parameters = copy.deepcopy(current_default_params)
                    sim_info = copy.deepcopy(current_sim_info)
                    sheets = current_sheets
                    data_type = current_data_type
                    st_name = current_st_name
                    recorders = current_recorders
                else:
                    if current_default_params != default_parameters:
                        raise ValueError(f"File {file_list[idx]} has different default parameters")
                    if current_sim_info != sim_info:
                        raise ValueError(f"File {file_list[idx]} has different sim_info")
                    if not np.array_equal(current_sheets, sheets):
                        raise ValueError(f"File {file_list[idx]} has different sheets")
                    if current_data_type != data_type:
                        raise ValueError(f"File {file_list[idx]} has different data_type")
                    if current_st_name != st_name:
                        raise ValueError(f"File {file_list[idx]} has different st_name")
                    if current_recorders != recorders:
                        raise ValueError(f"File {file_list[idx]} has different recorders")

            # add back results_dir and run_date to default_parameters and sim_info
            default_parameters['results_dir'] = results_dirs
            sim_info['run_date'] = run_dates

            # Write attributes to merged file
            f_merged.attrs['default_parameters'] = str(default_parameters)
            f_merged.attrs['sim_info'] = str(sim_info)
            f_merged.attrs['experimental_protocols'] = experimental_protocols
            f_merged.attrs['data_type'] = data_type
            f_merged.attrs['st_name'] = st_name
            f_merged.attrs['sheets'] = sheets
            f_merged.attrs['recorders'] = recorders

            # Process model subgroups
            for model_key in files[0].keys():
                model_subgroups = [f[model_key] for f in files]
                model_params = [eval(msg.attrs['parameters']) for msg in model_subgroups]
                for mp in model_params:
                    mp.pop('results_dir')
                if not all(mp == model_params[0] for mp in model_params):
                    raise ValueError("Model parameters differ across files")

                model_subgroup_merged = f_merged.create_group(model_key)
                model_subgroup_merged.attrs['parameters'] = str(model_params[0])

                # Process stimulus subgroups
                for stim_key in model_subgroups[0].keys():
                    stim_subgroups = [msg[stim_key] for msg in model_subgroups]
                    
                    # Check constant parameters
                    constant_params = stim_subgroups[0].attrs['constant_parameters']
                    if not all(np.array_equal(ssg.attrs['constant_parameters'], constant_params) for ssg in stim_subgroups):
                        raise ValueError("Constant parameters differ across files")
                    for constant_key in constant_params:
                        if not all(ssg.attrs[constant_key] == stim_subgroups[0].attrs[constant_key] for ssg in stim_subgroups):
                            raise ValueError(f"Constant parameter {constant_key} differs across files")

                    # Check varying parameters
                    varying_params = stim_subgroups[0].attrs['varying_parameters']
                    if not all(np.array_equal(ssg.attrs['varying_parameters'], varying_params) for ssg in stim_subgroups):
                        raise ValueError("Varying parameters differ across files")

                    different_key = None
                    different_param_dim = None
                    for varying_key_idx, varying_key in enumerate(varying_params):
                        values = [set(ssg.attrs[varying_key]) for ssg in stim_subgroups]
                        if not all(v == values[0] for v in values):
                            if different_key is not None:
                                raise ValueError("More than one varying parameter has different values across files")
                            if len(set.union(*values)) != sum(len(v) for v in values):
                                raise ValueError(f"Values for {varying_key} are not completely different across all files")
                            different_key = varying_key
                            different_param_dim = varying_key_idx

                    if different_key is None:
                        raise ValueError("No varying parameter has different values across files")

                    merged_stim_subgroup = model_subgroup_merged.create_group(stim_key)

                    # Copy attributes
                    for k in stim_subgroups[0].attrs.keys():
                        if k != different_key:
                            merged_stim_subgroup.attrs[k] = stim_subgroups[0].attrs[k]
                        else:
                            try: 
                                merged_stim_subgroup.attrs[k] = np.concatenate([ssg.attrs[k] for ssg in stim_subgroups])
                            except:
                                merged_stim_subgroup.attrs[k] = 'too large to be saved in attributes: saved in dataset'
                                merged_stim_subgroup.create_dataset(k, data=np.concatenate([ssg.attrs[k] for ssg in stim_subgroups], axis=different_param_dim), dtype=h5py.special_dtype(vlen=np.dtype('str')))

                    # Add merging dimension and sizes of each input dataset in that dimension
                    merged_stim_subgroup.attrs['merging_dimension'] = different_param_dim
                    merged_stim_subgroup.attrs['merging_dimension_name'] = different_key
                    merged_stim_subgroup.attrs['merging_input_sizes'] = [ssg['stimuli'].shape[different_param_dim] for ssg in stim_subgroups]

                    # Merge datasets
                    for sheet in sheets:
                        sheet_data = np.concatenate([ssg[sheet][:] for ssg in stim_subgroups], axis=different_param_dim)
                        if f_merged.attrs['data_type'] == 'spiketrains':
                            merged_stim_subgroup.create_dataset(sheet, data=sheet_data, dtype=h5py.special_dtype(vlen=np.dtype('float')))
                        else:
                            merged_stim_subgroup.create_dataset(sheet, data=sheet_data)

                    stimuli_idx_merged = np.concatenate([ssg['stimuli_idx'][:] + sum(len(s['stimuli'][:]) for s in stim_subgroups[:i]) for i, ssg in enumerate(stim_subgroups)], axis=different_param_dim)
                    merged_stim_subgroup.create_dataset('stimuli_idx', data=stimuli_idx_merged)

                    stimuli_merged = np.concatenate([ssg['stimuli'][:] for ssg in stim_subgroups])
                    merged_stim_subgroup.create_dataset('stimuli', data=stimuli_merged)

                logging.info(f'Merged {stim_key} in {model_key}')

    logging.info(f'Successfully merged {len(file_list)} files into {output_file}')

def get_stimuli_and_response_datasets(file_path, model_key, stim_key, sheet, mean_over_trials=False, select_trials=None, response_indices=None):
    """
    Retrieve pairs of stimuli and responses from an HDF5 file for a given model and stimulus key.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing the data.
    model_key : str
        Key identifying the model within the HDF5 file.
    stim_key : str
        Key identifying the stimulus within the model group.
    sheet : str
        Name of the sheet from which to extract responses.
    mean_over_trials : bool, optional
        If True, average the responses over trials. Default is False.
    select_trials : list of int, optional
        Specific trials to select from the responses. If None, all trials are used. Default is None.
    response_indices : list of int, optional
        Indices specifying which responses (and stimuli) to extract. If None, all responses (and stimuli) are extracted. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - stimuli : numpy.ndarray
            The extracted stimuli data.
        - responses : numpy.ndarray
            The extracted responses data, optionally averaged over trials.

    """
    with h5py.File(file_path, 'r') as f:
        # Navigate to the specific subgroup
        subgroup = f[model_key][stim_key]
        
        # Get the stimuli index and responses
        stimuli_idx = subgroup['stimuli_idx'][:]
        trial_dim = subgroup.attrs['trial_dim']

        # Use advanced indexing to directly get the stimuli
        if response_indices is not None:
            stimuli = subgroup['stimuli'][tuple(response_indices)]
        else:
            matching_idxs = np.take(stimuli_idx, 0, axis=trial_dim)
            stimuli = subgroup['stimuli'][matching_idxs]

        # Load only specific responses if response_indices is provided
        if response_indices is not None:
            response_slices = [slice(None)] + response_indices
            responses = subgroup[sheet][tuple(response_slices)]
        else:
            responses = subgroup[sheet][:]

        if select_trials is not None:
            responses = np.take(responses, select_trials, axis=trial_dim)

        # Optionally average over trial
        if mean_over_trials:
            responses = responses.mean(axis=trial_dim)
        
        return stimuli, responses

def print_dataset_content(file_path, dataset_path):
    """
    Print the content of a specific dataset in an HDF5 file.

    Parameters
    ----------
    file_path : str
                Path to the HDF5 file
    dataset_path : str  
                Path to the dataset within the HDF5 file
    """
    with h5py.File(file_path, 'r') as f:
        if dataset_path in f:
            dataset = f[dataset_path]
            print(dataset.shape)
            print(f"Content of dataset: {dataset_path}")
            
            # Check if the dataset contains variable-length data
            if h5py.check_dtype(vlen=dataset.dtype) == np.dtype('float'):
                print("Variable-length float data:")
                for i, row in enumerate(dataset):
                    print(f"  Row {i}: {row}")
            else:
                print(dataset[:])
        else:
            print(f"Dataset {dataset_path} not found in the file.")

def explore_hdf5(file_path):
    """
    Explore the entire structure of an HDF5 file, printing all groups, datasets, attributes, and top-level attributes.

    Parameters
    ----------
    file_path : str
                Path to the HDF5 file
    """
    def print_attrs(name, obj):
        print(f"Object: {name}")
        for key, val in obj.attrs.items():
            print(f"  Attribute: {key} = {val}")

    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
        print_attrs(name, obj)

    with h5py.File(file_path, 'r') as f:
        print("\nTop-level attributes:")
        for key, val in f.attrs.items():
            print(f"  {key} = {val}")

        print("File Structure:")
        f.visititems(print_structure)

def get_structure_of_hdf5(hdf5_file):
        """
        Get the structure of an HDF5 file.

        Parameters
        ----------
        hdf5_file : h5py.File
                    An open HDF5 file

        Returns
        -------
        dict
            A dictionary representing the structure of the HDF5 file.
            The keys are the names of the groups or datasets, and the values are dictionaries
            with the following keys:
            - 'type': A string indicating the type ('group' or 'dataset').
            - 'attributes': A list of attribute names.
        """
        structure = {}
        def visit(name, obj):
            obj_type = 'group' if isinstance(obj, h5py.Group) else 'dataset'
            structure[name] = {'type': obj_type, 'attributes': list(obj.attrs.keys())}
        hdf5_file.visititems(visit)
        return structure

def get_hdf5_group_list_of_attributes(file_path, group_name=None):
    """
    Get a list of attributes for a given HDF5 group or the file itself.

    Parameters
    ----------
    file_path : str
                Path to the HDF5 file
    group_name : str, optional
                Name of the group within the HDF5 file. If None, attributes of the file itself are returned.

    Returns
    -------
    list
        A list of attribute names.
    """
    with h5py.File(file_path, 'r') as hf:
        if group_name is None:
            return list(hf.attrs.keys())
        else:
            return list(hf[group_name].attrs.keys())

def get_hdf5_group_attribute(file_path, group_name, attribute_name):
    """
    Get the value of a specific attribute for a given HDF5 group or the file itself.

    Parameters
    ----------
    file_path : str
                Path to the HDF5 file
    group_name : str
                Name of the group within the HDF5 file. If None, the attribute of the file itself is returned.
    attribute_name : str
                Name of the attribute to retrieve

    Returns
    -------
    object
        The value of the specified attribute.
    """
    with h5py.File(file_path, 'r') as hf:
        if group_name is None:
            return hf.attrs[attribute_name]
        else:
            return hf[group_name].attrs[attribute_name]
        
def compare_hdf5_structure(file1, file2):
    """
    Compare the structure of two HDF5 files and return the differences.

    Parameters
    ----------
    file1 : str
                Path to the first HDF5 file
    file2 : str
                Path to the second HDF5 file

    Returns
    -------
    dict
        A dictionary containing the differences between the two HDF5 files.
        The keys are the names of the groups or datasets that differ, and the values are dictionaries
        with the following keys:
        - 'status': A string indicating the status of the difference ('missing in file1', 'missing in file2', 'different').
        - 'missing_attributes': A dictionary containing the attributes that are missing in either file1 or file2, 
          with the group they belong to.
    """

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        structure1 = get_structure_of_hdf5(f1)
        structure2 = get_structure_of_hdf5(f2)

    differences = {}
    all_keys = set(structure1.keys()).union(set(structure2.keys()))
    for key in all_keys:
        if key not in structure1:
            differences[key] = {'status': 'missing in file1', 'missing_attributes': {key: structure2[key]['attributes']}}
        elif key not in structure2:
            differences[key] = {'status': 'missing in file2', 'missing_attributes': {key: structure1[key]['attributes']}}
        else:
            if structure1[key] != structure2[key]:
                missing_in_file1 = list(set(structure2[key]['attributes']) - set(structure1[key]['attributes']))
                missing_in_file2 = list(set(structure1[key]['attributes']) - set(structure2[key]['attributes']))
                differences[key] = {
                    'status': 'different',
                    'missing_attributes': {
                        'missing_in_file1': {key: missing_in_file1},
                        'missing_in_file2': {key: missing_in_file2}
                    }
                }

    return differences

