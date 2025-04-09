import ctypes
import time
import sklearn


import os
import sys

# Change the working directory to the script's directory
sys.path.insert(0, r"C:\Users\reidj\Dropbox\dphil\programming\spikes\spikes")

print(f"Current working directory: {os.getcwd()}")

# Flags from the Windows API
ES_CONTINUOUS      = 0x80000000  # Informs the system that the state being set should remain in effect until the next call.
ES_SYSTEM_REQUIRED = 0x00000001  # Forces the system to be in the working state by resetting the system idle timer.
ES_DISPLAY_REQUIRED= 0x00000002  # Forces the display to be on by resetting the display idle timer.

# Prevent sleep: This call tells Windows to keep the system and display awake.
ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
)




if __name__ == "__main__":
    """
    This is a running list of all items that need to be defined in the main function:

    """
# OKAY THIS WHOLE THING ISN"T RIGHT - SINGLE EQUATION vs EQUATIONS?

    ############# IMPORTS ###############
    print("Imports")
    from brian2 import *
    from network import *
    from input import *
    from run import *
    from analysis import *
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import code
    print("imports done")
    """
    This is a running list of all items that need to be defined in the main function:
    """

    def monitoring_setup(monitor_manager, network):
        """
        This function defines what will be done during the monitor set up part of the main function
        """
        #################### 🛠 Monitor Creation 🛠 ####################
        # Create spike monitors dynamically
        for i in range(N_LAYERS+1):  # Excitatory monitors
            monitor_manager.create_monitor("spike", i, True)
        for i in range(1, N_LAYERS+1):  # Inhibitory monitors (1-4)
            monitor_manager.create_monitor("spike", i, False)


    def analysis_function(monitor_manager: MonitorManager, network: Network):
        """
        This function defines what will be done during the analysis part of the main function
        remember you can use nonlocal to start editing variables - not sure why you would
        """

        # To extract synapse info
        def synapse_spec_info(synapse_specs):
            synapse_groups = synapse_specs.synapse_objects
            synapse_spec_info_data = {}
            for key, synapse in synapse_groups.items():
                synapse_spec_info_data[synapse[0].name] = {
                    "source": synapse[0].source.name,
                    "target": synapse[0].target.name,
                    "i": np.array(synapse[0].i),
                    "j": np.array(synapse[0].j),
                    "w": np.array(synapse[0].w),
                    "d": np.array(synapse[0].delay),
                }
            return synapse_spec_info_data

        spike_monitors = {f"exc_{i}": monitor_manager.monitors[(i, "spike", True)] for i in range(N_LAYERS+1)}
        spike_monitors.update({f"inh_{i}": monitor_manager.monitors[(i, "spike", False)] for i in range(1, N_LAYERS+1)})
        print("Monitors created:")
        n_neurons = exc_neuron_specs.length
        print("doing the index dict")
        index_dict = {i+1: n_neurons * i for i in range(N_LAYERS)}

        connection_list = [[] for _ in range(exc_neuron_specs.length * N_LAYERS)]
        w_list = [[] for _ in range(exc_neuron_specs.length * N_LAYERS)]
        d_list = [[] for _ in range(exc_neuron_specs.length * N_LAYERS)]
        print("getting synapse set")
        dicts = [efe_synapse_specs.synapse_objects, ele_synapse_specs.synapse_objects, ebe_synapse_specs.synapse_objects]
        synapses_set = [value[0] for d in dicts for value in d.values()]
        connection_list = [[] for _ in range(exc_neuron_specs.length * exc_neuron_specs.length * N_LAYERS)]
        w_list = [[] for _ in range(exc_neuron_specs.length * exc_neuron_specs.length * N_LAYERS)]
        d_list = [[] for _ in range(exc_neuron_specs.length * exc_neuron_specs.length * N_LAYERS)]
        index_dict = {i+1: (exc_neuron_specs.length * exc_neuron_specs.length) * i for i in range(N_LAYERS)}
        for synapses in synapses_set:
            name = synapses.source.name
            layer_i = int(name[-1])
            synapse_name = synapses.name
            print(f"Getting connections for {synapse_name}")
            if "f" in synapse_name:
                layer_j = layer_i + 1
            elif "l" in synapse_name:
                layer_j = layer_i
            elif "b" in synapse_name:
                layer_j = layer_i - 1
            print(f"doing the zip")
            for i, j, w, d in zip(synapses.i, synapses.j, synapses.w, synapses.delay):
                connection_list[i + index_dict[layer_i]].append(j + index_dict[layer_j])
                w_list[i + index_dict[layer_i]].append(w)
                d_list[i + index_dict[layer_i]].append((d/ms)) # convert to absolute value in terms of ms for saving
        
        
        print("getting spike trains")
        # now get the list of spike trains for each excitatory layer and combine them
        spike_trains = [list(spike_monitors[f"exc_{i}"].spike_trains().values()) for i in range(1,N_LAYERS+1)]
        # list of lists in seconds
        spike_train_list = []
        for spike_train in spike_trains:
            spike_train = [spikes / second for spikes in spike_train]
            spike_train_list = spike_train_list + spike_train
        spike_train_list = [np.array(st) for st in spike_train_list] # This strips off the units
        print(spike_train_list[0][:10])
        print("turning into NEO")
        #now turn this into a neo object
        import neo
        import os
        import nixio
        import quantities as pq
        from neo.io import NixIO
        for monitor_name in spike_monitors.keys():
            print(f"{monitor_name} for {spike_monitors[monitor_name].source}")
        
        # Set up times
        from neo.core import SpikeTrain, Segment, Block
        t_start = (NO_EPOCHS * epoch_length) / second
        t_stop = t_start + ((TEST_STIMULUS_LENGTH * NUM_INPUTS * NO_TEST_EPOCHS) / second)
        t_start = t_start * pq.s
        t_stop = t_stop * pq.s

        # get the date
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%d_%m_%Y")
        print(date_str)

        # Create Neo objects:
        block = Block(name="Spike Trains")
        segment = Segment(name=f"Spike Trains_{date_str}")
        block.segments.append(segment)

        # Add spike trains to the segment
        for neuron_id, times in enumerate(spike_train_list):
            times = times * pq.s
            st = SpikeTrain(times = times, t_start=t_start, t_stop=t_stop)
            st.annotations["neuron_id"] = neuron_id
            st.annotations["forward_connections"] = connection_list[neuron_id]
            st.annotations["weights"] = w_list[neuron_id]
            st.annotations["delays"]  = d_list[neuron_id] * pq.s
            segment.spiketrains.append(st)


        # Add the synapse info to the segment
        synapse_specs = [efe_synapse_specs,ele_synapse_specs,ebe_synapse_specs,eli_synapse_specs,ile_synapse_specs]
        synapse_info = {synapse_spec.name : synapse_spec_info(synapse_spec) for synapse_spec in synapse_specs}
        for obj in network.objects:
            if hasattr(obj, "name"):
                if obj.name == "efe_0":
                    print("got EFE_0")
                    synapse_info["efe_0"] = {
                        "source": obj.source.name,
                        "target": obj.target.name,
                        "i":np.array(obj.i),
                        "j": np.array(obj.j),
                        "w": np.array(obj.w),
                        "d": np.array(obj.delay),
                    }
        block.annotations["synapse_info"] = synapse_info
        os.makedirs("results", exist_ok=True)
        print("Saving all data to NEO file")
        neo_filename = os.path.join("results", f"network_data{date_str}.nix")

        io = NixIO(filename=neo_filename, mode="ow")
        io.write(block)
        io.close()
        #Now save the synapse info to a file

        #################### 📊 Spike Heatmap Extraction 📊 ####################
        # Extract and display all spike heatmaps
        print("doing spike counts")
        spike_counts = {
            name: extract_spike_heatmap(monitor, width=image_height if "exc_0" in name else exc_neuron_specs.length if "exc" in name else inh_neuron_specs.length, n_filters=num_filters if "exc_0" in name else 1, is_input= True if "exc_0" in name else False)
            for name, monitor in spike_monitors.items()
        }
        from sklearn.metrics import mutual_info_score
        #################### 📦 Binned Spike Processing  📦 ####################
        print("doing binned spike counts")
        binned_spike_counts = {
            name: extract_binned_spike_heatmap(monitor, TEST_STIMULUS_LENGTH, start_time, start_time + NUM_INPUTS*TEST_STIMULUS_LENGTH, width=image_height if "exc_0" in name else exc_neuron_specs.length if "exc" in name else inh_neuron_specs.length, n_filters=num_filters if "exc_0" in name else 1)
            for name, monitor in spike_monitors.items()
        }
        stimulus_labels = np.arange(NUM_INPUTS)
        for key, item in binned_spike_counts.items():
            print(f"{key}: {item.shape}")
            if not key == "exc_0":
                data = np.squeeze(item)
                print(f"Data shape: {data.shape}")
                data_flat = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
                print(f"Data flat shape: {data_flat.shape}")
                #Now doing MI:
                mi_values = np.array([mutual_info_score(stimulus_labels, data_flat[:, i]) for i in range(data_flat.shape[1])])
                plt.hist(mi_values, bins=20, edgecolor="black")
                plt.xlabel("Mutual Information (bits)")
                plt.ylabel("Number of Neurons")
                plt.title("Distribution of MI Across Neurons")
                #save the figure
                plt.savefig(f"visualised_spikes\{NO_EPOCHS}_{key}_mi.png")
                plt.show()                
        print("doing binned display")
        for name, binned_heatmap in binned_spike_counts.items():
            plot_binned_heatmaps(binned_heatmap, f"visualised_spikes\e{NO_EPOCHS}_{name}_b_htmp.png", name)
        print("doing display")
        for name, heatmap in spike_counts.items():
            display_spike_heatmap(heatmap, f"visualised_spikes\e{NO_EPOCHS}_{name}_u_htmp.png", is_input= True if "exc_0" in name else False)

        #################### 🛠 Interactive Debugging Shell 🛠 ####################
        print("Entering interactive shell...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("Sleep settings restored.")

    def analysis_function_2(monitor_manager: MonitorManager, network: Network):
        """
        This function defines what will be done during the analysis part of the main function
        remember you can use nonlocal to start editing variables - not sure why you would
        """
        # To extract synapse info

        spike_monitors = {f"exc_{i}": monitor_manager.monitors[(i, "spike", True)] for i in range(N_LAYERS+1)}
        spike_monitors.update({f"inh_{i}": monitor_manager.monitors[(i, "spike", False)] for i in range(1, N_LAYERS+1)})
        print("Monitors created:")
        
        #################### 📊 Spike Heatmap Extraction 📊 ####################
        # Extract and display all spike heatmaps
        print("doing spike counts")
        spike_counts = {
            name: extract_spike_heatmap(monitor, width=image_height if "exc_0" in name else exc_neuron_specs.length if "exc" in name else inh_neuron_specs.length, n_filters=num_filters if "exc_0" in name else 1, is_input= True if "exc_0" in name else False)
            for name, monitor in spike_monitors.items()
        }
        # Save spike counts to files
        # Create directory if it doesn't exist
        os.makedirs("untrained_counts", exist_ok=True)
        
        # Save each spike count array as numpy file
        for name, spike_count in spike_counts.items():
            save_path = os.path.join("untrained_counts", f"{NO_EPOCHS}_{name}_spike_counts.npy")
            np.save(save_path, spike_count)
            print(f"Saved spike counts for {name} to {save_path}")

        # Also save as a single compressed file with all spike counts
        all_counts_path = os.path.join("untrained_counts", f"all_spike_counts_{NO_EPOCHS}.npz")
        np.savez_compressed(all_counts_path, **spike_counts)
        print(f"Saved all spike counts to {all_counts_path}")
        #################### 📦 Binned Spike Processing  📦 ####################
        #################### 🛠 Interactive Debugging Shell 🛠 ####################
        print("Entering interactive shell...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("Sleep settings restored.")
    ##########GET INPUTS ############
    print("running main")
    equations_container = EquationsContainer()
    network = Network()

    N_LAYERS = 4  # Number of layers to create
    NO_EPOCHS = 0
    NO_TEST_EPOCHS = 10
    STIMULUS_LENGTH = 100 * ms
    NUM_INPUTS = 8
    epoch_length = STIMULUS_LENGTH * NUM_INPUTS
    STORAGE = None
    DESCRIBE_NETWORK = False
    TEST_STIMULUS_LENGTH = 250 * ms
    # Constants
    start_time = NO_EPOCHS * epoch_length
    end_time = start_time + TEST_STIMULUS_LENGTH * NUM_INPUTS * NO_TEST_EPOCHS
    num_bins = NUM_INPUTS
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

    # Define neuron specifications:
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
    print("Generating Inputs")
    _3d_poisson_rates = (
        gen_inputs()
    )  # This has the shape num_images, neuron_size, neuron_size, num_filters
    print("Creating Network")
    timed_input, poisson_neurons = create_network(
        network,
        N_LAYERS,
        exc_neuron_specs,
        inh_neuron_specs,
        RADII,
        AVG_NO_CONNECTIONS,
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
        stimulus_exposure_time=STIMULUS_LENGTH,
        stimulus_exposure_time_test=TEST_STIMULUS_LENGTH,
        _3d_poisson_rates=_3d_poisson_rates,
        no_epochs=NO_EPOCHS,
        no_test_epochs=NO_TEST_EPOCHS,
    )
    print("initialising monitor manager")
    monitor_manager = MonitorManager(network)
    namespace = {
        "timed_input": timed_input,
    }
    ####################    MONITOR NETWORK   ####################

    ####################    TRAIN NETWORK   ####################
    defaultclock.dt = 0.1 * ms
    #run_training(network, namespace, STIMULUS_LENGTH, NUM_INPUTS, no_epochs=NO_EPOCHS)
    monitoring_setup(monitor_manager, network)
    run_testing_epochs(network, namespace, TEST_STIMULUS_LENGTH, NUM_INPUTS, no_testing_epochs=NO_TEST_EPOCHS)
    #analysis_function(monitor_manager, network)
    #enter interactive mode
    code.interact(local=globals())
    print("End of main")
