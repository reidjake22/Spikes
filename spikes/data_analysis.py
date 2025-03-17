from neo.io import NixIO
import quantities as pq
import numpy as np
import tqdm
import time
import ray
import elephant
from datetime import datetime
import logging
import ctypes
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG)

ES_CONTINUOUS      = 0x80000000  # Informs the system that the state being set should remain in effect until the next call.
ES_SYSTEM_REQUIRED = 0x00000001  # Forces the system to be in the working state by resetting the system idle timer.
ES_DISPLAY_REQUIRED= 0x00000002  # Forces the display to be on by resetting the display idle timer.

# Prevent sleep: This call tells Windows to keep the system and display awake.
ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
)

filename = "results/network_data27_02_2025.nix"
with NixIO(filename, mode="rw") as io:
    block = io.read_block()
print(block.segments[0].spiketrains[0].times)
print(f"Number of segments: {len(block.segments)}")
print(f"Number of spike_trains: {len(block.segments[0].spiketrains)}")

spiketrains = block.segments[0].spiketrains
metadata = {}
forward_connections = {}  # (source, target) -> delay
forward_sets = {}  # neuron -> set of targets
forward_weights = {} # (source, target) -> weight
side_connections = {} # (source, target) -> delay
side_sets = {} # neuron -> set of targets
side_weights = {} # (source, target) -> weight

n_neurons = 64**2
print(n_neurons)
# Build efficient lookup structures
print("building filtered lookup structures")
for i, values in tqdm.tqdm(enumerate(spiketrains),desc="building filtered lookup structures"):
    metadata[i] = values.annotations
    connections = values.annotations['forward_connections']
    delays = values.annotations['delays']
    connection_weights = values.annotations['weights']
    this_layer_start = (i//n_neurons) * n_neurons
    next_layer_start = this_layer_start + n_neurons
    # Initialize a filtered set for this neuron
    forward_sets[i] = set()
    side_sets[i] = set()
    # Filter connections with weight threshold
    for j, (target, weight) in enumerate(zip(connections, connection_weights)):
        if weight > 0.5:
            if target >= next_layer_start:  # Only keep connections with weights >= 0.5 and in the next layer
                forward_sets[i].add(target)
                forward_connections[(i, target)] = delays[j]
                forward_weights[(i, target)] = weight
            elif target >= this_layer_start: # Only keep connections with weights >= 0.5 and in the same layer
                side_sets[i].add(target)
                side_connections[(i, target)] = delays[j]
                side_weights[(i, target)] = weight
print("done")
print(f"forward len {len(forward_connections)}")
print(f"side len {len(side_connections)}")

triples = []
sparse_triples = []
alpha = 3 * pq.s

# Find triples efficiently with set operations
print("Finding triples with weight-filtered connections...")
for neuron1 in tqdm.tqdm(metadata.keys(), desc="Processing neurons"):
    targets_of_neuron1 = forward_sets[neuron1]
    for neuron2 in targets_of_neuron1:
        t1_2 = forward_connections[(neuron1, neuron2)]
        shared_targets = targets_of_neuron1.intersection(side_sets[neuron2])
        for neuron3 in shared_targets:
            t1_3 = forward_connections[(neuron1, neuron3)]
            t2_3 = side_connections[(neuron2, neuron3)]
            triples.append((neuron1, neuron2, neuron3))
            if abs(t1_2 + t2_3 - t1_3) < alpha:
                sparse_triples.append((neuron1, neuron2, neuron3))

print(f"Found {len(triples)} triples, {len(sparse_triples)} of which are sparse triples")

def add_silent_gaps(spiketrains, segment_duration=250*pq.ms, gap_duration=50*pq.ms):
    modified_spiketrains = []
    total_spikes = sum(len(st) for st in spiketrains)
    print(f"Processing {len(spiketrains)} spike trains with {total_spikes} total spikes")
    print(f"Adding {gap_duration} gaps every {segment_duration}")
    for st in tqdm.tqdm(spiketrains, desc="Processing spike trains"):
        times_ms = st.rescale('ms').magnitude
        segment_duration_ms = segment_duration.rescale('ms').magnitude
        gap_duration_ms = gap_duration.rescale('ms').magnitude
        segments = np.floor(times_ms / segment_duration_ms).astype(int)
        offsets = segments * gap_duration_ms
        new_times_ms = times_ms + offsets
        total_segments = int(np.ceil(st.t_stop.rescale('ms').magnitude / segment_duration_ms))
        new_t_stop_ms = st.t_stop.rescale('ms').magnitude + (total_segments * gap_duration_ms)
        original_units = st.units
        new_st = st.duplicate_with_new_data(
            (new_times_ms * pq.ms).rescale(original_units), 
            t_stop=(new_t_stop_ms * pq.ms).rescale(original_units)
        )
        new_st.annotations.update(st.annotations)
        modified_spiketrains.append(new_st)
    original_duration = max(st.t_stop for st in spiketrains)
    modified_duration = max(st.t_stop for st in modified_spiketrains)
    print(f"Processing complete: Duration extended from {original_duration} to {modified_duration}")
    return modified_spiketrains

modified_spikes = add_silent_gaps(block.segments[0].spiketrains)

test_data = []

def shift_spiketrain(spiketrain, shift):
    new_times = spiketrain.times - shift
    new_times = new_times[new_times >= spiketrain.t_start]
    shifted_st = spiketrain.duplicate_with_new_data(new_times, t_start=spiketrain.t_start, t_stop=spiketrain.t_stop)
    return shifted_st

for i, triple in enumerate(sparse_triples):
    if i % 500 == 0:
        print(f"Processing triple {i}/{len(sparse_triples)}")
    index_1_2 = metadata[triple[0]]['forward_connections'].index(triple[1])
    index_1_3 = metadata[triple[0]]['forward_connections'].index(triple[2])
    index_2_3 = metadata[triple[1]]['forward_connections'].index(triple[2])
    t1_2 = metadata[triple[0]]['delays'][index_1_2] / 1000
    t1_3 = metadata[triple[0]]['delays'][index_1_3] / 1000
    t2_3 = metadata[triple[1]]['delays'][index_2_3] / 1000
    st1 = modified_spikes[triple[0]]
    if len(st1) == 0:
        continue
    st2 = modified_spikes[triple[1]]
    if len(st2) == 0:
        continue
    st3 = modified_spikes[triple[2]]
    if len(st3) == 0:
        continue
    shifted_1 = st1
    shifted_2 = shift_spiketrain(st2, t1_2)
    shifted_3 = shift_spiketrain(st3, (t1_3 + max(t1_2 + t2_3 - t1_3, 0 * pq.s)))
    test_data.append({'data': [shifted_1, shifted_2, shifted_3]})

print(f"Processed {len(test_data)} triples")

# Shut down any existing Ray instance first
ray.shutdown()
# Initialize Ray with default memory settings
ray.init(num_cpus=3, logging_level=logging.INFO)

@ray.remote
class ProgressActor:
    def __init__(self, total_tasks):
        self.total = total_tasks
        self.completed = 0
        self.start_time = time.time()

    def update(self):
        self.completed += 1
        if self.completed % 10 == 0 or self.completed == self.total:
            elapsed = time.time() - self.start_time
            tasks_per_sec = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.completed) / tasks_per_sec if tasks_per_sec > 0 else "unknown"
            eta_str = str(eta) if isinstance(eta, str) else f"{eta:.2f} sec"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Completed {self.completed}/{self.total} ({self.completed/self.total*100:.1f}%) "
                  f"- Rate: {tasks_per_sec:.2f} tasks/sec - ETA: {eta_str}")
        return self.completed

    def get_completed(self):
        return self.completed

@ray.remote
def process_triple(data_idx, triple_data, progress_actor):
    try:
        import gc
        spike_triple = triple_data['data']
        outcome = elephant.spade.spade(
            spiketrains=spike_triple,
            bin_size=1*pq.ms,
            winlen=3,
            min_spikes=3,
            min_occ=3,
            max_spikes=3,
            min_neu=3,
            n_surr=50,
            dither=5*pq.ms
        )
        # Clean up memory before returning
        spike_triple = None
        gc.collect()
        
        ray.get(progress_actor.update.remote())
        return data_idx, outcome
    except Exception as e:
        print(f"ERROR [{data_idx}]: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"ERROR [{data_idx}]: {traceback.format_exc()}")
        ray.get(progress_actor.update.remote())
        return data_idx, None

def get_results(spike_data_list, batch_size=10):
    """
    Process spike data in batches to prevent memory issues.
    
    Args:
        spike_data_list: List of spike data dictionaries
        batch_size: Number of items to process in each batch
    """
    total_samples = len(spike_data_list)
    progress_actor = ProgressActor.remote(total_samples)
    print(f"Starting processing of {total_samples} samples in batches of {batch_size}")
    
    all_results = []
    start = time.time()
    
    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        print(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start}-{batch_end-1}")
        
        # Submit batch tasks
        batch_futures = []
        for idx in range(batch_start, batch_end):
            batch_futures.append(process_triple.remote(idx, spike_data_list[idx], progress_actor))
        
        # Get results for this batch
        batch_results = ray.get(batch_futures)
        all_results.extend(batch_results)
        
        # Save intermediate results
        valid_results = [result for result in all_results if result[1] is not None]
        temp_save_path = "results/processed_results_temp.pkl"
        with open(temp_save_path, 'wb') as f:
            pickle.dump(valid_results, f)
        
        # Force garbage collection between batches
        import gc
        gc.collect()
    
    end = time.time()
    print(f"Processing completed in {end-start:.2f} seconds")
    return all_results

# Main execution
try:
    results = get_results(test_data, batch_size=10)
    print("All processing completed successfully")
    
    # Save final results
    os.makedirs("results", exist_ok=True)
    final_save_path = "results/processed_results.pkl"
    valid_results = [result for result in results if result[1] is not None]
    print(f"Saving {len(valid_results)} valid results")
    with open(final_save_path, 'wb') as f:
        pickle.dump(valid_results, f)
    
    print("Results saved successfully")
except Exception as e:
    print(f"Error in main execution: {type(e).__name__}: {str(e)}")
    import traceback
    print(traceback.format_exc())
finally:
    print(ray.timeline())
    ray.shutdown()
    print("Ray shutdown complete")