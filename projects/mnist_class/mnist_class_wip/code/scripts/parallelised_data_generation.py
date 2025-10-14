
# Initialise Some stuff
import numpy as np
from numba import njit
@njit
def _fast_latency_and_count(neuron_idx, spike_times_ms, n_neurons):
    counts = np.zeros(n_neurons, dtype=np.int32)
    first_spike = np.full(n_neurons, 100.0, dtype=np.float32)
    for k in range(neuron_idx.shape[0]):
        i = neuron_idx[k]
        t = spike_times_ms[k]
        counts[i] += 1
        if t < first_spike[i]:
            first_spike[i] = t
    return first_spike, counts

from brian2 import *
import multiprocessing
import os
from glob import glob
import cv2
import sys
import pandas as pd
import time
sys.path.insert(0, r"/home/jake/Document/Spikes/spikes")
# now do local imports
from network import *
from input import *
from projects import *
from run import *

# Set up some directories
neuron_input_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/data/neuron_input_batch/"
conv_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/data/conv_mnist_batch/"

project = "mnist_class_wip"
local_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts"
config_dir = os.path.join(local_dir, os.pardir, os.pardir,"configs")
results_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials"
# set up equations & parameters
equations_container = EquationsContainer()
N_layers = 4
STIMULUS_LENGTH = 100 * ms
image_height, image_width, num_filters = 128, 128, 8
grid_size = 64  # For excitatory layers
off_stdp_model = """
            lambda_e: siemens
            alpha_C: 1
            alpha_D: 1
            tau_c: second
            tau_d: second
            w: 1
            plasticity: 1
            learning_rate: 1
            """
off_stdp_on_pre = """
            ge_post += lambda_e * w
            """
off_stdp_on_post = """"""
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
def create_initial_network(wsd_version_folder_path ):
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
        model=off_stdp_model,
        on_pre=off_stdp_on_pre,
        on_post=off_stdp_on_post,
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
        model=off_stdp_model,
        on_pre=off_stdp_on_pre,
        on_post=off_stdp_on_post,
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
        model=off_stdp_model,
        on_pre=off_stdp_on_pre,
        on_post=off_stdp_on_post,
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

    print(wsd_version_folder_path)
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
        storage_path=wsd_version_folder_path,
        use_timed_array=False,
    )
    synapse_specs_list = [
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    ]
    print(defaultclock.dt)
    final_layer_monitor = SpikeMonitor(exc_neuron_specs.neuron_groups[4], record=True, name="final_layer_monitor")
    network.add(final_layer_monitor)
    return network



def get_latency_and_sum(final_layer_monitor, n_neurons=4096):
    """
    Fast, JIT-compiled extraction of first-spike latency and spike count.
    """
    # Flattened spike arrays
    neuron_idx     = np.array(final_layer_monitor.i)
    spike_times_ms = np.array((final_layer_monitor.t / ms).astype(np.float32))
        # Call the compiled helper
    first_spike, counts = _fast_latency_and_count(
        neuron_idx,
        spike_times_ms,
        n_neurons
    )
    return first_spike, counts

class SimWrapper:
    def __init__(self, wsd_version: str):
        network = create_initial_network(os.path.join("/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/configs/network/", wsd_version))
        self.network = network
        self.wsd_version = wsd_version
        print("using wsd_version:", self.wsd_version)
        self.network.run(100*ms)
        device.build(clean=True, run=False, debug=False)
        self.device = get_device()

    def do_run(self, batch_start: int):
        # Runs in each process
        print(f"Running batch {batch_start//50}")
        training_data = "epoch_0_item_0"
        from brian2.devices import device_module
        device_module.active_device = self.device
        latency_list = np.zeros((100, 4096))  # Assuming 4096 neurons in the final layer
        no_spikes_list = np.zeros((100, 4096)) # Assuming 4096 neurons in the final layer
        conv_labels_list = np.zeros((100,))  # Assuming 100 samples in the batch
        cwd = os.getcwd()
        # Create relative path from absolute path
        rel_path = os.path.relpath(
            os.path.join(results_dir, "device_results", self.wsd_version, f"batch_{batch_start}"),
            cwd
        )

        for batch in range(2):
            neuron_inputs = generate_neuron_inputs_from_saved(conv_dir + f"batch_{batch_start//50}--{50}" + ".npy",
                                    r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/configs/input/mapping.npz")
            conv_labels = np.load(os.path.join(conv_dir, f"conv_labels_batch_{batch_start//50}--50.npy"))
            for i in range(50):
                if i % 10 == 0:
                    print(f"        batch {batch_start//50}: {i}/50")
                input_rates = neuron_inputs[i,:] * 6000 * Hz     
                self.device.run(run_args={self.network['p_0'].rates: input_rates},
                                results_directory=rel_path,)
                latency, no_spikes = get_latency_and_sum(self.network['final_layer_monitor'], n_neurons=4096)
                latency_list[i + batch*50, :] = latency
                no_spikes_list[i + batch*50, :] = no_spikes

            conv_labels_list[batch*50:(batch+1)*50] = conv_labels
        stim_ids = np.arange(batch_start, batch_start + 100)
        neuron_ids = np.arange(4096)
        stimulus_col = np.repeat(stim_ids, 4096)
        neuron_col = np.tile(neuron_ids, 100)
        latency_flat = latency_list.ravel()
        count_flat = no_spikes_list.ravel()
        metric_df = pd.DataFrame({
            'stimulus_id': stimulus_col,
            'neuron_id': neuron_col,
            'latency': latency_flat,
            'count': count_flat,
            'conv_label': np.repeat(conv_labels_list, 4096)
        })
        batch_dir = os.path.join(results_dir, self.wsd_version, f"batch_{batch_start}")
        os.makedirs(batch_dir, exist_ok=True)
        out_file = os.path.join(batch_dir, "metrics.parquet")
        metric_df.to_parquet(out_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
        )
        return batch_start


if __name__ == '__main__':
    set_device('cpp_standalone', build_on_run=False, debug=False)
    sim = SimWrapper(wsd_version="epoch_0_item_0")
    import multiprocessing
    total = len(range(0,60000,100))
    print(f"Total batches to process: {total}", flush=True)
    start_time = time.time()
    completed = 0
    n_processes = 4
    print(f"Using {n_processes} processes for parallel execution.", flush=True)
    with multiprocessing.Pool(processes=n_processes) as p:
        for i, bs in enumerate(p.imap_unordered(sim.do_run, range(0,60000,100)), start = 1):
            completed += 1
            elapsed = time.time() - start_time
            eta = (elapsed/completed) * (total - completed) if completed > 0 else 0
            print(f"Progress: [{completed}/{total}] {completed/total*100:.1f}% | "
                  f"Last finished: Batch {bs} | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"ETA: {eta:.1f}s", 
                  flush=True)
