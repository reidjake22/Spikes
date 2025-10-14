import os
import numpy as np
import glob

def save_wcd(synapse_specs_list, storage_path, epoch_folder):
    """
    Save weights and connectivity information from a list of synapse specification objects.
    
    Parameters:
    -----------
    synapse_specs_list : list
        List of SynapseSpecs objects containing synapse data to save
    storage_path : str
        Base directory path where to save the data
    epoch_no : int
        Current epoch number (for file organization)
    
    Returns:
    --------
    None
    """
    # Create epoch directory
    epoch_dir = os.path.join(storage_path, epoch_folder)
    os.makedirs(epoch_dir, exist_ok=True)
    
    saved_count = 0
    
    # Process each synapse specs object
    for specs in synapse_specs_list:
        # Access the synapse objects dictionary
        # Format is typically {layer_id: [(synapses, afferent_group, efferent_group), ...]}
        for layer, synapse_tuple in specs.synapse_objects.items():
            synapses, afferent_group, efferent_group = synapse_tuple
            
            # Create filename based on layer and connected groups
            filename = f"{layer}_{afferent_group.name}_{efferent_group.name}_synapses.npz"
            file_path = os.path.join(epoch_dir, filename)
            
            # Save the synapse data
            np.savez(
                file_path,
                w=synapses.w,
                i=synapses.i,
                j=synapses.j,
                delay=synapses.delay
            )
            
            saved_count += 1
            print(f"Saved {file_path}")
    
    print(f"Successfully saved {saved_count} synapse objects to {epoch_dir}")

def set_wcd(synapse_specs_list, storage_path, epoch_no):
    """
    Load synapse weights from storage and apply them directly to synapse objects.
    
    Parameters:
    -----------
    synapse_specs_list : list
        List of SynapseSpecs objects whose synapses will be updated
    storage_path : str
        Base directory path where the weights are stored
    epoch_no : int
        Epoch number to load from
    
    Returns:
    --------
    int
        Number of synapse objects updated
    """
    # Construct epoch directory path
    epoch_dir = os.path.join(storage_path, f"epoch_{epoch_no}")
    
    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")
    
    # Track updates
    updated_count = 0
    
    # Process each synapse specs object
    for specs in synapse_specs_list:
        # Look for files matching this specs name
        spec_files = glob.glob(os.path.join(epoch_dir, f"{specs.name}_*_synapses.npz"))
        
        if not spec_files:
            print(f"Warning: No files found for synapse type {specs.name}")
            continue
        
        # Process each synapse tuple in the specs
        for layer, synapse_tuple in specs.synapse_objects.items():
            synapses, afferent_group, efferent_group = synapse_tuple
            
            # Create filename pattern to match
            filename = f"{specs.name}_{layer}_{afferent_group.name}_{efferent_group.name}_synapses.npz"
            file_path = os.path.join(epoch_dir, filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: No data file found for {specs.name} layer {layer} {afferent_group.name}->{efferent_group.name}")
                continue
            
            # Load the data
            try:
                data = np.load(file_path)
                
                # Set the weights - this is the key operation
                synapses.w = data['w']
                synapses.i = data['i']
                synapses.j = data['j']
                synapses.delay = data['delay']
                updated_count += 1
                print(f"Updated weights for {specs.name} layer {layer} {afferent_group.name}->{efferent_group.name}")
            except Exception as e:
                print(f"Error loading data from {file_path}: {str(e)}")

    print(f"Successfully updated {updated_count} synapse objects with weights from {epoch_dir}")
    return updated_count


