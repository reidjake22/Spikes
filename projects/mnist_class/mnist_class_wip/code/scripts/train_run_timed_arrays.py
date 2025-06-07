# %%
from brian2 import *
set_device('cpp_standalone', build_on_run=False)
import multiprocessing
total = multiprocessing.cpu_count()
print(total)
reserve = 2
print(total-reserve)
import numpy as np
import os
from glob import glob
import cv2
import sys
sys.path.insert(0, r"/home/jake/Document/Spikes/spikes")
from network import *
from input import *
from projects import *
from run import *
filter_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/configs/input/filters"
conv_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/data/conv_mnist_batch/"
neuron_input_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/data/neuron_input_batch/"
prefs.devices.cpp_standalone.openmp_threads = total-reserve
equations_container = EquationsContainer()
project = "mnist_class_wip"
local_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts"
config_dir = os.path.join(local_dir, os.pardir, os.pardir,"configs")
# Set up parameters
N_layers = 4
STIMULUS_LENGTH = 100 * ms
image_height, image_width, num_filters = 128, 128, 8
grid_size = 64  # For excitatory layers
RADII = {
    "efe": {1: 8, 2: 12, 3: 16},
    "ele": {1: 2, 2: 2, 3: 2, 4: 2},
    "ebe": {2: 8, 3: 8, 4: 8},
    "eli": {1: 2, 2: 2, 3: 2, 4: 2},
    "ile": {1: 4, 2: 4, 3: 4, 4: 4},
}

AVG_NO_CONNECTIONS = {
    "efe": {0: 50, 1: 100, 2: 100, 3: 100},
    "ele": {1: 10, 2: 10, 3: 10, 4: 10},
    "ebe": {1: 10, 2: 10, 3: 10, 4: 10},
    "eli": {1: 10, 2: 10, 3: 10, 4: 10},
    "ile": {1: 30, 2: 30, 3: 30, 4: 30},
}

# Function to extract spikes per image time segment
def extract_spikes_per_image(spike_monitor, image_duration, n_images, width=64):
    """
    Extract spikes for each image time segment
    
    Parameters:
    - spike_monitor: Brian2 SpikeMonitor object
    - image_duration: Duration per image in seconds
    - n_images: Number of images shown
    - width: Width of the layer (assuming square)
    
    Returns:
    - List of spike count arrays for each image time segment
    """
    spike_heatmaps = []
    
    for i in range(n_images):
        # Calculate time window for this image
        t_start = i * image_duration 
        t_end = (i + 1) * image_duration
        
        # Filter spikes in this time window
        mask = (spike_monitor.t >= t_start*second) & (spike_monitor.t < t_end*second)
        image_spikes_t = spike_monitor.t[mask]
        image_spikes_i = spike_monitor.i[mask]
        
        # Count spikes per neuron in this window
        neuron_count = width * width
        spike_counts = np.zeros(neuron_count)
        
        for neuron_idx in image_spikes_i:
            spike_counts[neuron_idx] += 1
            
        # Reshape to grid and append to results
        spike_heatmaps.append(spike_counts.reshape(width, width))
    
    return spike_heatmaps


def run_training_batch(start_batch, start_folder, end_folder ):
    network = Network()

    print("Defining Excitatory Neurons")
    exc_neuron_specs = NeuronSpecs(
        neuron_type="e",
        length=64,
        cm=500 * pF,
        g_leak=25 * nS,
        v_threshold=-53 * mV,
        v_reset=-57 * mV,
        v_rest=-74 * mV,
        v_reversal_e=0 * mV,
        v_reversal_i=-70 * mV,
        v_reversal_a=-90 * mV,
        sigma=0.015 * mV,
        t_refract=2 * ms,
        tau_m=20 * ms,
        tau_ee=2 * ms,
        tau_ie=5 * ms,
        tau_a=80 * ms,
    )
    print("Defining Inhibitory Neurons")
    inh_neuron_specs = NeuronSpecs(
        neuron_type="i",
        length=32,
        cm=214 * pF,
        g_leak=18 * nS,
        v_threshold=-53 * mV,
        v_reset=-58 * mV,
        v_rest=-82 * mV,
        v_reversal_e=0 * mV,
        v_reversal_i=-70 * mV,
        sigma=0.015 * mV,
        t_refract=2 * ms,
        tau_m=12 * ms,
        tau_ei=2 * ms,
        tau_ii=5 * ms,
    )
    print("Defining Synapse Specifications")
    print("Defining EFE")
    efe_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="f",
        name="efe",
        lambda_e=30 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    print("Defining ELE")
    ele_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="l",
        name="ele",
        lambda_e=20 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    print("Defining EBE")
    ebe_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="b",
        name="ebe",
        lambda_e=20 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    print("Defining ELI")
    eli_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["excit_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["excit_non_stdp_on_pre"],
        type="l",
        name="eli",
        lambda_e=20 * nS,
    )
    print("Defining ILE")
    ile_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["inhib_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["inhib_non_stdp_on_pre"],
        type="l",
        name="ile",
        lambda_i=30 * nS,
    )

    print(os.path.join(config_dir, "network", start_folder))
    print("Creating Network")
    create_network(
        network,
        4,
        exc_neuron_specs,
        inh_neuron_specs,
        RADII,
        AVG_NO_CONNECTIONS,
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
        storage="load",
        storage_path=os.path.join(config_dir, "network", start_folder),
    )

    print(defaultclock.dt)
    import time
    NO_BATCHES = 10
    batch_size = 50 
    input_layer = exc_neuron_specs.neuron_groups[0]
    input_timed_array = [] 
    stimulus_time = 250*ms
    for batch in range(NO_BATCHES):
        current_batch = start_batch + batch
        convolved_images_location = conv_dir + f"batch_{current_batch}--{50}" + ".npy"
        neuron_inputs = generate_neuron_inputs_from_saved(convolved_images_location,
                                    r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/configs/input/mapping.npz")
        print("neuron_inputs shape:", neuron_inputs.shape)

        for item_idx in range(batch_size):
            print(defaultclock.t)
            inputs = neuron_inputs[item_idx]
            input_rates = inputs * 6000 * Hz
            print("Running network for batch", batch, "item", item_idx)
            input_timed_array.append(input_rates)
    timed_array = TimedArray(input_timed_array, dt = stimulus_time)
    print("Running network")
    #print(input_layer.namespace)
    # ta = input_layer.namespace['timed_array']
    # print(ta)

    network.run(stimulus_time*len(input_timed_array), report='text', report_period=100*ms)
    device.build(clean=True, run=False, debug=True)
    device.run()


    print("Saving network")
    synapse_specs_list = [
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    ]
    save_wcd(synapse_specs_list,
             os.path.join(config_dir, "network"),
             end_folder)
batch_size = 50
for epoch in range(3):
    for start in range (0, 60000, 500):
        if not (epoch == 0 and start == 0):
            device.reinit()
            device.activate(build_on_run=False)
            defaultclock.dt = 0.1*ms    # re‐set dt if non‐default
        start_batch = start // batch_size
        start_folder = f"epoch_{epoch}_item_{start}"
        end_folder = f"epoch_{epoch}_item_{start + 500}"
        print(f"Running training from items {start} to {start+500} (batches {start_batch} to {start_batch+20})")
        run_training_batch(start_batch, start_folder, end_folder)
        print(f"Completed training batch from {start} to {start + 500}")

