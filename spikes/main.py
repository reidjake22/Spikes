<<<<<<< HEAD
############# IMPORTS ###############
from brian2 import *
from network import *
from input import *
from run import *
from analysis import *
from tensorflow.keras.datasets import mnist
from PIL import Image
import orjson
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import pickle
from datetime import datetime


##########GET INPUTS ############
def gen_brian_inputs():
    """
    Generates 3D Poisson input rates using Gabor filters for a set of images.
    This function performs the following steps:
    1. Defines parameters for Gabor filters including wavelengths, scaling factors, orientations, phase offsets, and aspect ratios.
    2. Creates Gabor filters using the specified parameters.
    3. Generates permutations of image identifiers.
    4. Loads and processes images from the specified path.
    5. Converts images to greyscale and normalizes them.
    6. Generates 3D Poisson input rates from the processed images using the Gabor filters.
    Returns:
        np.ndarray: A 3D array of Poisson input rates generated from the images.
    """

=======
import ctypes
import time
import sklearn


import os
import sys

# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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
>>>>>>> jakes_working_repo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import code

<<<<<<< HEAD
    lambdas = [0.8]  # Wavelengths
    betas = [1]  # Scaling factor for bandwidth
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Orientations
    psis = [0, np.pi]  # Phase offsets
    gammas = [0.5]  # Aspect ratio
    size = 6  # Gabor filter sizes
    gabor_filters = GaborFilters(size, lambdas, betas, thetas, psis, gammas)

    from itertools import product

    permutations = list(product(["c", "v"], repeat=3))
    image_path = "data/3N2P/"
    # Create an np array with 8 images and each size 124x124:
    images = np.zeros((8, 128, 128))
    image_no = 0
    print("analysing data")
    for t, l, r in permutations:
        print(f"t: {t}, l: {l}, r: {r}")
        image_paths = image_path + f"t{t}_l{l}_r{r}.jpg"
        print(image_paths)
        image = Image.open(image_paths)
        greyscale = image.convert("L")
        greyscale = np.array(greyscale) / 255.0
        images[image_no] = greyscale
        image_no += 1

    _3d_poisson_inputs = generate_3d_poisson_rates_from_filters(
        images,
        gabor_filters,
        neuron_size=64,
        image_size=128,
    )
    # This has the shape num_images, neuron_size, neuron_size, num_filters

    return _3d_poisson_inputs


if __name__ == "__main__":

    ########## MAIN LOGIC ############
    # Create equations and network
    equations_container = EquationsContainer()
    network = Network()

    # Define Constants
    # import file from weights/initial_setup.json as DATA
    with open("weights/initial_setup.json", "rb") as file:
        DATA = orjson.loads(file.read())

    N_LAYERS = 4  # Number of layers to create
    NO_EPOCHS = 60
    STIMULUS_LENGTH = 100 * ms
    NUM_INPUTS = 8
    STORAGE = None  # It worked but should be falsy so be concerned - I think I check whether it's None not False
    DESCRIBE_NETWORK = False
    input_lambda_e = 30 * nS
    TEST_STIMULUS_LENGTH = 250 * ms
    RESTORE = (False,)
    STORE = (False,)
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
=======
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
        """"
        Now the neohdf5 stuff
        The structure we want is:
        block:
                                    segment 1 ...   segment n
        recording channel group
    
        """
    ##########GET INPUTS ############
    print("running main")
    equations_container = EquationsContainer()
    network = Network()

    N_LAYERS = 4  # Number of layers to create
    NO_EPOCHS = 60
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
>>>>>>> jakes_working_repo
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
<<<<<<< HEAD

=======
    print("Defining Inhibitory Neurons")
>>>>>>> jakes_working_repo
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
<<<<<<< HEAD

    # Define synapse specifications:
    # Define synapse specifications:
=======
    print("Defining Synapse Specifications")
    print("Defining EFE")
>>>>>>> jakes_working_repo
    efe_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="f",
<<<<<<< HEAD
        lambda_e=30 * nS,
        lambda_a=6 * nS,
=======
        name="efe",
        lambda_e=30 * nS,
>>>>>>> jakes_working_repo
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
<<<<<<< HEAD
        learning_rate=0.04,  # NO IT ?aint
    )

=======
        learning_rate=0.04,
    )
    print("Defining ELE")
>>>>>>> jakes_working_repo
    ele_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="l",
<<<<<<< HEAD
        lambda_e=20 * nS,
        lambda_a=6 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    ebe_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="b",
        lambda_e=20 * nS,
        lambda_a=6 * nS,
=======
        name="ele",
        lambda_e=20 * nS,
>>>>>>> jakes_working_repo
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
<<<<<<< HEAD

    eli_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["inhib_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["inhib_non_stdp_on_pre"],
        type="l",
        lambda_i=30 * nS,
    )
    ile_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["excit_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["excit_non_stdp_on_pre"],
        type="l",
        lambda_e=20 * nS,
        lambda_a=6 * nS,
    )

    _3d_poisson_rates = (
        gen_brian_inputs()
    )  # This has the shape num_images, neuron_size, neuron_size, num_filters
    absolute_3d_poisson_rates = np.abs(_3d_poisson_rates)
    # Create Neuron Groups:
    create_neuron_groups(network, N_LAYERS, exc_neuron_specs, inh_neuron_specs)

    # Create Synapses:
    create_synapse_groups(
        network,
        N_LAYERS,
        RADII,
        AVG_NO_CONNECTIONS,
        exc_neuron_specs,
        inh_neuron_specs,
=======
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
    absolute_3d_poisson_rates = np.abs(_3d_poisson_rates)
    print("Creating Network")
    timed_input, poisson_neurons = create_network(
        network,
        N_LAYERS,
        exc_neuron_specs,
        inh_neuron_specs,
        RADII,
        AVG_NO_CONNECTIONS,
>>>>>>> jakes_working_repo
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
<<<<<<< HEAD
        storage=STORAGE,
        data=DATA,
        store=STORE,
        restore=RESTORE,
    )
    ####################    Sort inputs   ####################
    print("Generating inputs")

    # Got to make sure this is defined globally - can it be added to the network/included globally?
    epoch_length = STIMULUS_LENGTH * NUM_INPUTS

    visualise_poisson_inputs(absolute_3d_poisson_rates)

    train_input, test_input, poisson_neurons = wire_input_layer_brian(
        network,
        exc_neuron_specs,
        absolute_3d_poisson_rates,
        beta=6000,
        radius=2,
        avg_no_neurons=AVG_NO_CONNECTIONS["efe"][0],
        epoch_length=epoch_length,
        stimulus_exposure_time=STIMULUS_LENGTH,
        stimulus_exposure_time_test=TEST_STIMULUS_LENGTH,
        input_lambda_e=input_lambda_e,
        storage=STORAGE,
        data=DATA,
        store=STORE,
    )
    # Save initial weights of each synapse for each layer
    initial_weights = {}
    synapse_specs = [
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    ]
    for synapse_spec in synapse_specs:
        synapse_type = (
            f"{synapse_spec.recent_a}{synapse_spec.type}{synapse_spec.recent_e}"
        )
        initial_weights[synapse_type] = {}
        for layer, synapse_group in synapse_spec.synapse_objects.items():
            initial_weights[synapse_type][layer] = {
                "i": np.array(synapse_group[0].i),
                "j": np.array(synapse_group[0].j),
                "w": np.array(synapse_group[0].w),
                "delay": np.array(synapse_group[0].delay),
            }

    os.makedirs("results/full_monty", exist_ok=True)
    with open("results/full_monty/initial_weights.pkl", "wb") as file:
        pickle.dump(initial_weights, file)
    namespace = {
        "input_lambda_e": input_lambda_e,
        "timed_input": train_input,
        "epoch_length": epoch_length,
    }
    # spike_counter = NeuronGroup(N=64 * 64, model="dv/dt=0*Hz:1")
    # network.add(spike_counter)
    # synapses_to_counter = Synapses(
    #     poisson_neurons, spike_counter, model="w:1", on_pre="v+=w"
    # )
    # i_indices = DATA["i_0"]
    # j_indices = DATA["j_0"]
    # synapses_to_counter.connect(i=i_indices, j=j_indices)
    # synapses_to_counter.w = 1
    # network.add(synapses_to_counter)
    # counter_monitor = StateMonitor(
    #     source=spike_counter, dt=STIMULUS_LENGTH, record=True, variables="v"
    # )
    # network.add(counter_monitor)

    # if STORAGE:
    #     print(f"storing the following in storage {list(STORAGE.keys())}")
    #     store_synapses(
    #         STORAGE,
    #         N_LAYERS,
    #         exc_neuron_specs,
    #         inh_neuron_specs,
    #         "weights",
    #         "initial_setup.json",
    #     )

    print(running_network)

    ####################    MONITOR NETWORK   ####################
    network_monitors = Monitors(network, N_LAYERS)
    # network_monitors.setup_poisson_monitors("spike")
    network_monitors.setup_excitatory_monitors([1, 2, 3, 4], "spike")
    network_monitors.toggle_monitoring([1], "spike", enable=False)

    ####################    TRAIN NETWORK   ####################
    defaultclock.dt = 0.1 * ms
    run_training(network, namespace, STIMULUS_LENGTH, NUM_INPUTS, no_epochs=NO_EPOCHS)
    # recorded_times = counter_monitor.t
    # cumulative_activity = counter_monitor.v
    # period_activity = np.zeros((64 * 64, len(recorded_times)))
    # for i in range(64 * 64):
    #     for j in range(len(recorded_times)):
    #         period_activity[i, j] = (
    #             cumulative_activity[i, j] - cumulative_activity[i, j - 1]
    #             if j > 0
    #             else cumulative_activity[i, j]
    #         )
    # print(cumulative_activity[:, 7])
    # num_images = 8

    # # Iterate over each image and create a heatmap
    # for i in range(num_images + 1):
    #     plt.figure(figsize=(6, 6))
    #     plt.title(f"Neuron Activity Heatmap for Image {i+1}")

    #     # Reshape period_activity for the current image into a 64x64 grid
    #     activity_grid = period_activity[:, i].reshape(64, 64)

    #     # Create the heatmap
    #     plt.imshow(activity_grid, cmap="hot", interpolation="nearest")
    #     plt.colorbar(label="Activity Level")
    #     plt.xlabel("Neuron Index (X)")
    #     plt.ylabel("Neuron Index (Y)")
    #     # Save the heatmap to a file
    #     output_dir = "results/misc_data/no_spikes_per_image"
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_path = os.path.join(output_dir, f"img_{i+1}.png")
    #     plt.savefig(output_path)
    #     plt.show()
    ####################    TEST NETWORK   ####################
    namespace["timed_input"] = test_input
    namespace["epoch_length"] = TEST_STIMULUS_LENGTH * NUM_INPUTS
    timed_input = test_input
    epoch_length = TEST_STIMULUS_LENGTH * NUM_INPUTS
    # binned_spikes = network_monitors.bin_poisson_spikes(NUM_INPUTS, STIMULUS_LENGTH)

    # for i in range(NUM_INPUTS):
    #     plt.figure()
    #     spike_counts = binned_spikes[i, :]
    #     bins = [0, 1, 2, 3, 4, 5]
    #     hist, bin_edges = np.histogram(spike_counts, bins=bins + [np.inf])

    #     plt.bar(bins, hist, align="center", tick_label=["0", "1", "2", "3", "4", "5+"])
    #     for j in range(len(hist)):
    #         plt.text(bins[j], hist[j], str(hist[j]), ha="center", va="bottom")

    #     plt.xlabel("Number of spikes")
    #     plt.ylabel("Number of neurons")
    #     plt.title(f"Histogram of spikes for image {i}")
    #     plt.ylim(0, max(hist) * 1.1)
    #     plt.show()

    spike_monitor_0 = SpikeMonitor(poisson_neurons, record=True)
    spike_monitor_1 = SpikeMonitor(exc_neuron_specs.neuron_groups[1], record=True)
    spike_monitor_2 = SpikeMonitor(exc_neuron_specs.neuron_groups[2], record=True)
    spike_monitor_3 = SpikeMonitor(exc_neuron_specs.neuron_groups[3], record=True)
    spike_monitor_4 = SpikeMonitor(exc_neuron_specs.neuron_groups[4], record=True)
    monitors = {
        0: spike_monitor_0,
        1: spike_monitor_1,
        2: spike_monitor_2,
        3: spike_monitor_3,
        4: spike_monitor_4,
    }
    for layer, monitor in monitors.items():
        network.add(monitor)
    print("running testing epoch")
    run_testing_epoch(
        network_monitors, network, namespace, TEST_STIMULUS_LENGTH, NUM_INPUTS
    )
    print("analysing data")
=======
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
>>>>>>> jakes_working_repo

    data = network_monitors.the_full_monty(
        directory="results", filename="spike_data_26th_november_2024"
    )

<<<<<<< HEAD
    def get_all_relevant_data(
        Network,
        monitors,
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    ):
        """
        Get all relevant data from the network and store it in a dictionary.
        Returns:
            dict: A dictionary containing all relevant data from the network.
            weights (dict): A dictionary containing the weights of all synapses.
            spikes (dict): A dictionary containing the spike times of all excitatory neurons contained within the monitors object.
        """
        weights_and_delays = {}
        synapses = {
            "efe": efe_synapse_specs,
            "ele": ele_synapse_specs,
            "ebe": ebe_synapse_specs,
            "eli": eli_synapse_specs,
            "ile": ile_synapse_specs,
        }
        for synapse_type_str, synapse_type_object in synapses.items():
            dictionary = {}
            synapse_groups = synapse_type_object.synapse_objects
            layer = 0
            for key, item in synapse_groups.items():
                dictionary[layer] = {
                    "i": np.array(item[0].i),
                    "j": np.array(item[0].j),
                    "w": np.array(item[0].w),
                    "delay": np.array(item[0].delay),
                }
                layer += 1
            weights_and_delays[synapse_type_str] = dictionary
        spikes = {}
        for key, item in monitors.items():
            spikes[key] = dict(item.spike_trains())
        return {"weights_and_delays": weights_and_delays, "spikes": spikes}

    # synapses = [efe_synapse_specs, ele_synapse_specs, ebe_synapse_specs, eli_synapse_specs, ile_synapse_specs]
    # weights_and_delays = {}
    # for synapse_type in synapses:
    #         dictionary = {}
    #         synapse_groups = synapse_type.synapse_objects
    #         dictionary["key"] = {
    #             "i": np.array(synapse_groups[0].i),
    #             "j": np.array(synapse_groups[0].j),
    #             "w": np.array(synapse_groups[0].w),
    #             "delay": np.array(synapse_groups[0].delay),
    #         }
    #         weights_and_delays[synapse_type] = dictionary
    # spikes = {}
    # for key, item in monitors.items():
    #     spikes[key] = dict(item.spike_trains())
    # print(type(spikes))

    data_in_network = get_all_relevant_data(
        network,
        monitors,
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    )
    print(type(data_in_network))
    os.makedirs("results/full_monty", exist_ok=True)
    with open("results/full_monty/dataset.pkl", "wb") as file:
        pickle.dump(data_in_network, file)

    for layer, monitor in monitors.items():
        print(f"number of spikes in layer {layer}: {len(monitor.i)}")
=======
    ####################    TRAIN NETWORK   ####################
    defaultclock.dt = 0.1 * ms
    run_training(network, namespace, STIMULUS_LENGTH, NUM_INPUTS, no_epochs=NO_EPOCHS)
    monitoring_setup(monitor_manager, network)
    run_testing_epochs(network, namespace, TEST_STIMULUS_LENGTH, NUM_INPUTS, no_testing_epochs=NO_TEST_EPOCHS)
    analysis_function(monitor_manager, network)
    #enter interactive mode
    code.interact(local=globals())
    print("End of main")
>>>>>>> jakes_working_repo
