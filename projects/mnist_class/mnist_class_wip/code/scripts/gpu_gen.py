import os
from glob import glob
import argparse
import numpy as np
import pandas as pd
from numba import njit
from brian2 import *
import brian2cuda
import sys
import time

# Add project path for imports
sys.path.insert(0, r"/home/jake/Document/Spikes/spikes")
from network import *
from input import *
from projects import *
from run import *

# === JIT helper for latency/count extraction ===
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

# === CONFIGURATION ===
noise_levels = [0, 5, 15, 30, 50]
train_test_splits = ['train', 'test']

# Directories
input_dir    = "/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/data/neuron_input_batch"
config_root = "/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/configs/network"
results_dir = "/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials"

# Network parameters
equations_container = EquationsContainer()
N_layers          = 4
STIMULUS_LENGTH   = 100*ms
image_height, image_width, num_filters = 128, 128, 8

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

off_stdp_model = '''
    lambda_e: siemens
    alpha_C: 1
    alpha_D: 1
    tau_c: second
    tau_d: second
    w: 1
    plasticity: 1
    learning_rate: 1
'''
off_stdp_on_pre = 'ge_post += lambda_e * w'
off_stdp_on_post = ''

# === GPU Setup ===
def setup_gpu(gpu_index, checkpoint):
    current_time = time.strftime("%Y%m%d_%H%M%S")
    build_dir = f"cuda_gpu{gpu_index}_cp{checkpoint}_{current_time}"
    
    set_device('cuda_standalone', build_on_run=False, directory=build_dir)
    prefs.devices.cuda_standalone.cuda_backend.gpu_id = gpu_index
    print(f"GPU {gpu_index} setup complete, build directory: {build_dir}")

def create_initial_network(wsd_version_folder_path):
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
    
    eli_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["excit_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["excit_non_stdp_on_pre"],
        type="l",
        name="eli",
        lambda_e=20 * nS,
    )
    
    ile_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["inhib_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["inhib_non_stdp_on_pre"],
        type="l",
        name="ile",
        lambda_i=30 * nS,
    )

    print(f"Creating Network from: {wsd_version_folder_path}")
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
    
    final_layer_monitor = SpikeMonitor(exc_neuron_specs.neuron_groups[4], record=True, name="final_layer_monitor")
    network.add(final_layer_monitor)
    return network

# === Network Compilation ===
def compile_model(checkpoint):
    print(f"Compiling model for checkpoint {checkpoint}...")
    checkpoint_path = os.path.join(config_root, f"epoch_0_item_{checkpoint}")
    network = create_initial_network(checkpoint_path)
    
    print("Running initial simulation and building...")
    network.run(STIMULUS_LENGTH)
    device.build(clean=True, run=False, debug=False)
    print("Model compilation complete!")
    return network

# === Simulation & Metric Saving ===
def run_and_save(network, batch_file, label_file, output_dir):
    neuron_inputs = np.load(batch_file)
    labels        = np.load(label_file)
    n_samples     = neuron_inputs.shape[0]
    n_neurons     = 4096

    latency_list = np.zeros((n_samples, n_neurons), dtype=np.float32)
    count_list   = np.zeros((n_samples, n_neurons), dtype=np.int32)

    for i in range(n_samples):
        if i % 10 == 0:
            print(f"    Sample {i}/{n_samples}")
        rates = neuron_inputs[i] * 6000 * Hz
        device.run(run_args={network['p_0'].rates: rates})
        m = network['final_layer_monitor']
        idx = np.array(m.i)
        tms = np.array(m.t/ms, dtype=np.float32)
        lat, cnt = _fast_latency_and_count(idx, tms, n_neurons)
        latency_list[i] = lat
        count_list[i]   = cnt

    # Flatten and save
    stim_ids = np.repeat(np.arange(n_samples), n_neurons)
    neu_ids  = np.tile(np.arange(n_neurons), n_samples)
    df = pd.DataFrame({
        'stimulus_id': stim_ids,
        'neuron_id':   neu_ids,
        'latency':     latency_list.ravel(),
        'count':       count_list.ravel(),
        'label':       np.repeat(labels, n_neurons)
    })
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(os.path.join(output_dir, 'metrics.parquet'), engine='pyarrow', compression='snappy', index=False)

def process_checkpoint(checkpoint, gpu_id, noise_levels_to_process, splits_to_process):
    print(f"=== Processing checkpoint {checkpoint} on GPU {gpu_id} ===")
    setup_gpu(gpu_id, checkpoint)
    network = compile_model(checkpoint)
    
    total_tasks = len(noise_levels_to_process) * len(splits_to_process)
    current_task = 0
    
    for noise in noise_levels_to_process:
        for split in splits_to_process:
            current_task += 1
            print(f"\n--- Task {current_task}/{total_tasks}: Noise {noise}, Split {split} ---")
            
            folder = os.path.join(input_dir, f'noise_{noise}', split)
            batches = sorted(glob(os.path.join(folder, 'neuron_input_batch_*.npy')))
            labels  = sorted(glob(os.path.join(folder, 'labels_*.npy')))
            
            print(f"Found {len(batches)} batch files")
            
            for batch_idx, (bfile, lfile) in enumerate(zip(batches, labels)):
                batch_name = os.path.splitext(os.path.basename(bfile))[0]
                out_dir = os.path.join(results_dir,
                                        f'epoch_0_item_{checkpoint}',
                                        f'noise_{noise}',
                                        split,
                                        batch_name)
                print(f"  Processing batch {batch_idx+1}/{len(batches)}: {batch_name}")
                run_and_save(network, bfile, lfile, out_dir)
    
    print(f"=== Completed checkpoint {checkpoint} on GPU {gpu_id} ===")

# === Main Execution ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process neural network checkpoint on specific GPU')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use (0, 1, 2, etc.)')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint to process (0, 500, 2500, 12500)')
    parser.add_argument('--noise', type=int, nargs='+', default=None, 
                       help='Noise levels to process (e.g., --noise 0 5 15). If not specified, processes all: 0 5 15 30 50')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'test'],
                       help='Data splits to process (default: train test)')
    
    args = parser.parse_args()
    
    # Use specified noise levels or default to all
    noise_to_process = args.noise if args.noise is not None else noise_levels
    
    print(f"Starting processing:")
    print(f"  GPU: {args.gpu}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Noise levels: {noise_to_process}")
    print(f"  Data splits: {args.splits}")
    
    start_time = time.time()
    process_checkpoint(args.checkpoint, args.gpu, noise_to_process, args.splits)
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.1f} seconds")