# this is the main module. I will use this to gate running scripts as I develop. My aim is to develop clear well documented readable code.

from brian2 import *
from network import *
from input import *
from tensorflow.keras.datasets import mnist


def input_example():
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
    num_images = 30
    dataset = train_images[:num_images]

    # Normalize the dataset
    dataset = dataset.astype(np.float32) / 255.0
    neuron_inputs = generate_inputs_from_filters(
        dataset,
        gabor_filters,
        neuron_size=14,
        image_size=28,
        num_total_pixels=201,
        radius=8,
        shape="circle",
    )
    return neuron_inputs


def visnet():
    network = Network()
    n_layers = 2  # Number of layers to create

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

    input_lambda_e = 0.1 * nS

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
    timed_input = generate_timed_input_from_input(input_example(), 500 * ms)

    input_synapses = wire_input_layer(network, exc_neuron_specs, timed_input)

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
*             ✨   BROUGHT TO YOU BY JAKE    ✨                       *
**********************************************************************
    """
    )
    print("\n\nbrought to you by Jake Reid\n\n")
    network.run(
        10000 * ms,
    )
    # DONE!


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
