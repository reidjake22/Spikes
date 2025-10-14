import numpy as np
from numba import njit
import argparse
import os
from glob import glob
import pandas as pd
import time
import multiprocessing
import sys
import logging
from brian2 import *
from brian2 import device
sys.path.insert(0, r"/home/jake/Document/Spikes/spikes")
from network import *
from input import *
from projects import *
from run import *

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
input_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/data/neuron_input_batch/"
results_dir = r"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials"
equations_container = EquationsContainer()
STIMULUS_LENGTH = 100 * ms

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
off_stdp_on_pre = "ge_post += lambda_e * w"
off_stdp_on_post = ""

def create_initial_network(wsd_version_folder_path):
    network = Network()

    exc_neuron_specs = NeuronSpecs(
        neuron_type="e", length=64, cm=500 * pF, g_leak=25 * nS,
        v_threshold=-53 * mV, v_reset=-57 * mV, v_rest=-74 * mV,
        v_reversal_e=0 * mV, v_reversal_i=-70 * mV, v_reversal_a=-90 * mV,
        sigma=0.015 * mV, t_refract=2 * ms, tau_m=20 * ms,
        tau_ee=2 * ms, tau_ie=5 * ms, tau_a=80 * ms,
    )
    
    inh_neuron_specs = NeuronSpecs(
        neuron_type="i", length=32, cm=214 * pF, g_leak=18 * nS,
        v_threshold=-53 * mV, v_reset=-58 * mV, v_rest=-82 * mV,
        v_reversal_e=0 * mV, v_reversal_i=-70 * mV, sigma=0.015 * mV,
        t_refract=2 * ms, tau_m=12 * ms, tau_ei=2 * ms, tau_ii=5 * ms,
    )
    
    efe_synapse_specs = SynapseSpecs(
        model=off_stdp_model, on_pre=off_stdp_on_pre, on_post=off_stdp_on_post,
        type="f", name="efe", lambda_e=30 * nS, alpha_C=0.5, alpha_D=0.5,
        tau_c=5 * ms, tau_d=5 * ms, learning_rate=0.04,
    )
    
    ele_synapse_specs = SynapseSpecs(
        model=off_stdp_model, on_pre=off_stdp_on_pre, on_post=off_stdp_on_post,
        type="l", name="ele", lambda_e=20 * nS, alpha_C=0.5, alpha_D=0.5,
        tau_c=5 * ms, tau_d=5 * ms, learning_rate=0.04,
    )
    
    ebe_synapse_specs = SynapseSpecs(
        model=off_stdp_model, on_pre=off_stdp_on_pre, on_post=off_stdp_on_post,
        type="b", name="ebe", lambda_e=20 * nS, alpha_C=0.5, alpha_D=0.5,
        tau_c=5 * ms, tau_d=5 * ms, learning_rate=0.04,
    )
    
    eli_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["excit_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["excit_non_stdp_on_pre"],
        type="l", name="eli", lambda_e=20 * nS,
    )
    

    ile_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["inhib_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["inhib_non_stdp_on_pre"],
        type="l", name="ile", lambda_i=30 * nS,
    )

    create_network(
        network, 4, exc_neuron_specs, inh_neuron_specs, RADII, AVG_NO_CONNECTIONS,
        efe_synapse_specs, ele_synapse_specs, ebe_synapse_specs,
        eli_synapse_specs, ile_synapse_specs,
        storage="load", storage_path=wsd_version_folder_path, use_timed_array=False,
    )
    
    final_layer_monitor = SpikeMonitor(exc_neuron_specs.neuron_groups[4], record=True, name="final_layer_monitor")
    network.add(final_layer_monitor)
    return network

def get_latency_and_sum(final_layer_monitor, n_neurons=4096):
    neuron_idx = np.array(final_layer_monitor.i)
    spike_times_ms = np.array((final_layer_monitor.t / ms).astype(np.float32))
    first_spike, counts = _fast_latency_and_count(neuron_idx, spike_times_ms, n_neurons)
    return first_spike, counts

class SimWrapper:
    def __init__(self, checkpoint: int, noise: int, name: str):
        wsd_version = f"epoch_0_item_{checkpoint}"
        network_path = f"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/configs/network/{wsd_version}"
        
        self.network = create_initial_network(network_path)
        self.checkpoint = checkpoint
        self.noise = noise
        self.wsd_version = wsd_version
        
        logger.info(f"Using checkpoint: {checkpoint}, noise: {noise}")
        self.network.run(STIMULUS_LENGTH)
        device.build(clean=True, run=False, debug=False, directory=name )
        self.device = get_device()

    def do_run(self, batch_start: int, split: str):
        from brian2.devices import device_module
        device_module.active_device = self.device

        # Load data for this noise level
        split_folder = os.path.join(input_dir, f'noise_{self.noise}', split)
        batch_file = os.path.join(split_folder, f'neuron_input_batch_{batch_start}.npy')
        label_file = os.path.join(split_folder, f'labels_{batch_start}.npy')
        
        logger.info(f"Processing checkpoint {self.checkpoint}'s batch no. {batch_start} for split {split} with noise {self.noise}")
        logger.debug(f"Batch file: {batch_file}")
        
        neuron_inputs = np.load(batch_file)
        labels = np.load(label_file)
        n_samples = neuron_inputs.shape[0]
        
        latency_list = np.zeros((n_samples, 4096))
        no_spikes_list = np.zeros((n_samples, 4096))
        
        for stimulus in range(n_samples):            
            input_rates = neuron_inputs[stimulus] * 6000 * Hz
            self.device.run(run_args={self.network['p_0'].rates: input_rates})
            latency, no_spikes = get_latency_and_sum(self.network['final_layer_monitor'])
            latency_list[stimulus] = latency
            no_spikes_list[stimulus] = no_spikes
        # Save results
        stim_ids = np.repeat(np.arange(n_samples), 4096)
        neu_ids = np.tile(np.arange(4096), n_samples)


        df = pd.DataFrame({
            'stimulus_id': stim_ids,
            'neuron_id': neu_ids,
            'latency': latency_list.ravel(),
            'count': no_spikes_list.ravel(),
            'label': np.repeat(labels, 4096)
        })
        
        batch_name = os.path.splitext(os.path.basename(batch_file))[0]
        out_dir = os.path.join(results_dir, self.wsd_version, f'noise_{self.noise}', split, batch_name)
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, 'metrics.parquet'), engine='pyarrow', compression='snappy', index=False)
        return batch_start

def _worker(batch_args):
    return sim.do_run(*batch_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data on CPU')
    parser.add_argument('--split', type=str, required=True, choices=['test', 'train'], help='Split to process: test or train')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint number to load')
    parser.add_argument('--noise', type=int, required=True, choices=[0, 5, 15, 30, 50])
    parser.add_argument('--processes', type=int, default=None, help='Number of processes (default: remaining CPUs)')
    parser.add_argument('--start_batch', type=int, default=0, help='Start batch index (default: 0)')
    parser.add_argument('--end_batch', type=int, default=None, help='End batch index (default: None, process all batches)')
    args = parser.parse_args()
    # Calculate available CPUs (total - 16 reserved for GPUs)
    total_cpus = 16  # Your actual CPU count
    gpu_cpus = 4     # Only 1 core per GPU (not 16!)
    available_cpus = total_cpus - gpu_cpus  # 12 cores available
    n_processes = args.processes if args.processes else available_cpus
    logger.debug(f"{args.split}, {args.start_batch}, {args.checkpoint}, {args.noise}")
    logger.debug(f"Total CPUs: {total_cpus}")
    logger.debug(f"Reserved for GPUs: {gpu_cpus}")
    logger.debug(f"Available for this job: {available_cpus}")
    logger.info(f"Using: {n_processes} processes")
    name = f"build_{args.split}_checkpoint_{args.checkpoint}_noise_{args.noise}_start_{args.start_batch}"
    logger.info(name)
    set_device('cpp_standalone', build_on_run=False, debug=False, directory=name)
    sim = SimWrapper(checkpoint=args.checkpoint, noise=args.noise, name=name)
    if args.split == 'train':
        if args.end_batch is not None:
            if args.end_batch > 1200:
                raise ValueError("End batch for training split cannot exceed 1200.")
        batch_starts = range(args.start_batch, 1200) if args.end_batch is None else range(args.start_batch, args.end_batch)
    if args.split == 'test':
        if args.end_batch is not None:
            if args.end_batch is not None and args.end_batch > 200:
                raise ValueError("End batch for test split cannot exceed 200.")
        batch_starts = range(args.start_batch, 200) if args.end_batch is None else range(args.start_batch, args.end_batch) 
    total_batches = len(batch_starts)
    start_time = time.time()
    completed = 0
    batch_args = [(batch_idx, args.split) for batch_idx in batch_starts]
    with multiprocessing.Pool(processes=n_processes) as p:
        for run_index, result in enumerate(p.imap_unordered(_worker, batch_args, chunksize=1)):
            if result is not None:
                completed += 1
                elapsed = time.time() - start_time
                eta = (elapsed/completed) * (total_batches - completed) if completed > 0 else 0
                logger.info(f"Progress: [{completed}/{total_batches}] {completed/total_batches*100:.1f}% | ")
                logger.info(f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                
    logger.info(f"Completed checkpoint {args.checkpoint}, noise {args.noise} in {time.time() - start_time:.1f}s")