# this is the main module. I will use this to gate running scripts as I develop. My aim is to develop clear well documented readable code.

from brian2 import *
from network import *
from input import *
from run import *
from tensorflow.keras.datasets import mnist


def input_example(num_inputs):
    # Exemplifies how to use the filter module
    import numpy as np

    lambdas = [2]  # Wavelengths
    betas = [1.5]  # Scaling factor for bandwidth
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Orientations
    psis = [0, np.pi]  # Phase offsets
    gammas = [0.5]  # Aspect ratio
    size = 128
    gabor_filters = GaborFilters(size, lambdas, betas, thetas, psis, gammas)

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Extract the first 30 images
    num_images = num_inputs
    dataset = train_images[:num_images]

    # Normalize the dataset
    dataset = dataset.astype(np.float32) / 255.0
    neuron_inputs = generate_inputs_from_filters(
        dataset,
        gabor_filters,
        neuron_size=14,
        image_size=28,
        num_total_pixels=30,
        radius=8,
        shape="circle",
    )
    return neuron_inputs

def three_dim_visualise_synapses(synapses: Synapses):
    Ns = len(synapses.source)
    num_pre_neurons = len(synapses.N_incoming_pre)
    len_pre = int(sqrt(num_pre_neurons))
    Nt = len(synapses.target)
    num_post_neurons = len(synapses.N_incoming_post)
    len_post = int(sqrt(num_post_neurons))
    s_i_column = synapses.i % len_pre
    s_i_row = synapses.i // len_pre
    s_j_column = synapses.j % len_post
    s_j_row = synapses.j // len_post
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        s_i_column, s_i_row, 0, c="blue", label="Pre-synaptic neurons", s=100
    )  # Blue circles
    ax.scatter(
        s_j_column, s_j_row, 1, c="red", label="Post-synaptic neurons", s=100
    )  # Red circles
    for x1, y1, x2, y2 in zip(x_pre, y_pre, x_post, y_post):
        ax.plot([x1, x2], [y1, y2], [0, 1], "-k", alpha=0.6)  # Black lines with transparency

    for x1, y1, x2, y2 in zip(x_pre, y_pre, x_post, y_post):
        ax.plot([x1, x2], [y1, y2], [0, 1], "-k", alpha=0.6)  # Black lines with transparency

    plt.show()

def visnet():
    network = Network()
    on_plasticity = True

    def toggle_plasticity(state):
        if not isinstance(state, bool):
            raise ValueError("State must be a boolean")
        global on_plasticity
        on_plasticity = state

    n_layers = 2  # Number of layers to create
    stimulus_length = 500 * ms
    num_inputs = 30
    
    # Currently length is stores as a parameter in the NeuronSpecs class
    # This is less than optimal, but it works for now

    # Define neuron specifications for excitatory neurons
    exc_neuron_specs = NeuronSpecs(
        neuron_type="excitatory",
        length=14,
        cm=500 * pF,
        g_leak=25 * nS,
        v_threshold=-53 * mV,
        v_reset=-57 * mV,
        v_rest=-74 * mV,
        v_reversal_e=0 * mV,
        v_reversal_i=-70 * mV,
        sigma=0.015 * mV,
        t_refract=2 * ms,  # NEED TO ADD THIS
        tau_m=20 * ms,
        tau_ee=2 * ms,
        tau_ie=5 * ms,
    )

    # Define neuron specifications for inhibitory neurons
    inh_neuron_specs = NeuronSpecs(
        neuron_type="inhibitory",
        length=7,
        cm=214 * pF,
        g_leak=18 * nS,
        v_threshold=-53 * mV,
        v_reset=-58 * mV,
        v_rest=-82 * mV,
        v_reversal_e=0 * mV,
        v_reversal_i=-70 * mV,
        sigma=0.015 * mV,
        tau_m=12 * ms,
        tau_ei=2 * ms,
        tau_ii=5 * ms,
    )

    # Define STDP synapse specifications
    stdp_synapse_specs = StdpSynapseSpecs(
        lambda_e=0.1 * nS,
        A_minus=0.1,
        A_plus=0.1,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=3 * ms,
        tau_d=5 * ms,
    )

    # Define non-STDP synapse specifications
    non_stdp_synapse_specs = NonStdpSynapseSpecs(
        lambda_e=0.1 * nS,
        lambda_i=0.1 * nS,
    )

    input_lambda_e = 1 * nS

    # Create Synapses
    create_neuron_groups(network, n_layers, exc_neuron_specs, inh_neuron_specs)
    # Create Synapses
    create_synapse_groups(
        network,
        n_layers,
        exc_neuron_specs,
        inh_neuron_specs,
        stdp_synapse_specs,
        non_stdp_synapse_specs,
    )
    # Sort inputs
    print("Generating inputs")
    # Got to make sure this is defined globally - can it be added to the network/included globally

    neuron_inputs = input_example(num_inputs)

    neuron_inputs.visualise()

    timed_input = generate_timed_input_from_input(neuron_inputs, stimulus_length)
    epoch_length = stimulus_length * num_inputs
    input_synapses = wire_input_layer(
        network,
        exc_neuron_specs,
        timed_input,
        epoch_length=epoch_length,
    )

    print(
        r"""

**********************************************************************
*                                                                    *
*   █████   ██    ██  ███    ██  ███    ██ █████ ███    ██  ██████   *
*   ██   ██ ██    ██  ████   ██  ████   ██  ██   ████   ██ ██        *
*   █████   ██    ██  ██ ██  ██  ██ ██  ██  ██   ██ ██  ██ ██    ██  *
*   ██   ██ ██    ██  ██  ██ ██  ██  ██ ██  ██   ██  ██ ██ ██     █  *
*   ██   ██  ██████   ██   ████  ██   ████ █████ ██   ████   █████   *
*                                                                    *
*                                                                    *
*   ██      ██ ███████ ██████ ██     ██   █████  █████    ██    ██   *
*   ████   ██ ██         ██   ██     ██  ██   ██ ██   ██  ██  ██     *
*   ██ ██  ██ █████      ██   ██  █  ██  ██   ██ █████    ████       *
*   ██  ██ ██ ██         ██   ██ ███ ██  ██   ██ ██   ██  ██  ██     *
*   ██   ████ ███████    ██    ███ ███    █████  ██   ██  ██    ██   *
*                                                                    *
*             ✨   BROUGHT TO YOU BY OCTNAI    ✨                     *
**********************************************************************
    """
    )
    # SETUP MONITORS:

    monitors = Monitors(network, n_layers)
    # monitors.setup_monitors([1], "voltage")  # Cant monitor voltage of input layer
    monitors.setup_excitatory_monitors([1], "spike")
    monitors.toggle_monitoring(
        [2], "spike", enable=False
    )  # GONNA NEED TO SORT THAT TOO!
    namespace = {
        "input_lambda_e": input_lambda_e,
        "timed_input": timed_input,
        "epoch_length": epoch_length,
    }
    # TRAIN NETWORK:

    run_training(network, namespace, stimulus_length, num_inputs, no_epochs=1)
    # TEST NETWORK:
    run_testing_epoch(monitors, network, namespace, stimulus_length, num_inputs)
    print("analysing data")
    monitors.animate_spike_heatmap(
        1, "spike", num_inputs, stimulus_length, exc_neuron_specs.length
    )
    layer_1_synapses = 


if __name__ == "__main__":
    # When run from the command line we gate what is then run using the sys module

    import sys

    # The first argument is the script name

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "p1":
            input_example()
        elif command == "p2":
            visnet()
    else:
        print(
            """select what programme to run:
              fashion: project1 - looks at the fashion mnist
              """
        )
