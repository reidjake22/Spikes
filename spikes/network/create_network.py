"""
Module Name: create_network.py
----------------------------------------------------

Purpose:
--------
    This module provides functions to create and wire neuron and synapse
groups using Brian2 in a manner akin to a VisNet model.

Variables:
----------
    None

Functions:
----------
    create_neuron_groups:
        Generates n layers of excitatory and inhibitory neurons, 
    wiring them up according to the neuron specifications provided.

    create_synapse_groups:
        Takes a VisNet Neuron model and creates synapses between the layers of neurons
    according to the synapse specifications provided.

    wire_input_layer:
        creates an input layer of poisson neurons and wires them to the first
    excitatory layer of the network.

Classes:
--------
    None


Example Usage:
--------------
    TODO - add examples

Notes:
------
    When defining function parameters always follow the given order:
        Network,
        Layer,
        Neurons,
        Synapses,
        Specifications

    TODO - add flexibility in specifying radius
    TODO - add flexibility in specifying the number of neurons
    TODO - add flexibility in specifying the number of synapses
    TODO - add lateral and backward connections
"""

# Import the necessary modules
from brian2 import *
from .neurons import NeuronSpecs
from .synapses import SynapseSpecs
from input import *


def create_neuron_groups(
    network: Network,
    n_layers: int,
    exc_neuron_specs: NeuronSpecs,
    inh_neuron_specs: NeuronSpecs,
) -> None:
    """
    Overview:
    --------
        Generates n layers of excitatory and inhibitory neurons,
    wiring them up according to the neuron specifications provided.

    Details:
    -------
        It iterates over a range generated on n_layers, indexing layers from 1.
    So for a range(3) it will create layers 1, 2, and 3.
        This means we can define the poissongroup or input layer as layer 0.
        For each layer it calls the create_neurons method of the NeuronSpecs class.
    Parameters:
    ----------
        network (Network):
            The Brian2 network object to add the neurons to.
        n_layers (int):
            Number of layers in the network.
        exc_neuron_specs (NeuronSpecs)
            Specifications for the excitatory neuron group.
        inh_neuron_specs (NeuronSpecs)
            Specifications for the inhibitory neuron group.

    Returns:
    --------
        None

    -------
        We want to add flexibility in specifying the number of neurons and synapses,
    and whether we do back and lateral.
    """

    # Iterate over each layer and create neurons based on their types
    print(f"Creating {n_layers} neuron layers")
    print("---------------------------------")
    for layer in range(1, n_layers + 1):
        print(f"creating layer {layer}")
        print("- - - - - - - - - - - - -")

        # Create excitatory and inhibitory neuron groups for each layer
        exc_neuron_specs.create_neurons(layer, target_network=network)
        inh_neuron_specs.create_neurons(layer, target_network=network)


def create_synapse_groups(
    network,
    n_layers,
    radii,
    avg_no_neurons,
    exc_neuron_specs: NeuronSpecs,
    inh_neuron_specs: NeuronSpecs,
    efe_synapse_specs: SynapseSpecs,
    ele_synapse_specs: SynapseSpecs,
    ebe_synapse_specs: SynapseSpecs,
    eli_synapse_specs: SynapseSpecs,
    ile_synapse_specs: SynapseSpecs,
) -> None:
    """
    Overview:
    --------
    Takes a VisNet Neuron model and creates synapses between the layers of neurons
    according to the synapse specifications provided.

    Details:
    -------

    Parameters:
    -----------
    network (Network):
        The Brian2 network object to add the synapses to.
    n_layers (int):
        Number of layers in the network.
    exc_neuron_specs (NeuronSpecs):
        Specifications for the excitatory neuron group.
    inh_neuron_specs (NeuronSpecs):
        Specifications for the inhibitory neuron group.
    Returns:
    --------
        None

    Raises:
    -------
        None

    Example Usage:
    --------------
    TODO

    Notes:
    ------
    TODO Add the input synapse stuff in here

    """
    print("point 3")
    print(efe_synapse_specs.pre_point)
    # Iterate over each layer and create synapses based on their types | Can defo do this a lot faster! define a function, also makes the data and storage stuff more maleable
    print(f"creating {n_layers} synapse layers")
    for layer in range(1, n_layers + 1):
        print(f"creating synapses for layer {layer}")
        # Create efe synapses for all layers except the last
        if not layer == n_layers:
            # create efe synapses
            print(f"\r*** creating efe synapses for layer {layer} ***", flush=True)
            efe_synapse_specs.create_synapses(
                layer,
                exc_neuron_specs,
                exc_neuron_specs,
                target_network=network,
            )
            print(f"\r*** connecting efe synapses for layer {layer} ***", flush=True)
            efe_synapse_specs.connect_synapses(
                layer,
                radius=radii["efe"][layer],
                avg_no_neurons=avg_no_neurons["efe"][layer],
            )

        # create ele synapses
        print(f"\r*** creating ele synapses for layer {layer} *** ", flush=True)
        ele_synapse_specs.create_synapses(
            layer,
            exc_neuron_specs,
            exc_neuron_specs,
            target_network=network,
        )
        ele_synapse_specs.connect_synapses(
            layer,
            radius=radii["ele"][layer],
            avg_no_neurons=avg_no_neurons["ele"][layer],
        )
        # Create ebe synapses for all layers except the first
        if not layer == 1:
            # create ebe synapses
            print(f"\r*** creating ebe synapses for layer {layer}", flush=True)
            ebe_synapse_specs.create_synapses(
                layer,
                exc_neuron_specs,
                exc_neuron_specs,
                target_network=network,
            )
            ebe_synapse_specs.connect_synapses(
                layer,
                radius=radii["ebe"][layer],
                avg_no_neurons=avg_no_neurons["ebe"][layer],
            )

        # create E-I synapses
        print(f"\r*** creating eli synapses for layer {layer} *** ", flush=True)
        eli_synapse_specs.create_synapses(
            layer,
            exc_neuron_specs,
            inh_neuron_specs,
            target_network=network,
        )
        eli_synapse_specs.connect_synapses(
            layer,
            radius=radii["eli"][layer],
            avg_no_neurons=avg_no_neurons["eli"][layer],
        )

        # create I-E synapses
        print(f"\r*** creating ile synapses for layer {layer} *** ", flush=True)
        ile_synapse_specs.create_synapses(
            layer,
            inh_neuron_specs,
            exc_neuron_specs,
            target_network=network,
            debug=False,
        )
        ile_synapse_specs.connect_synapses(
            layer,
            radius=radii["ile"][layer],
            avg_no_neurons=avg_no_neurons["ile"][layer],
        )


def wire_input_layer_brian(
    network: Network,
    exc_neuron_specs: NeuronSpecs,
    _3d_poisson_rates,
    beta,
    radius,
    avg_no_neurons,
    stimulus_exposure_time,
    stimulus_exposure_time_test,
    no_epochs,
    no_test_epochs,
):
    """
    Creates an input layer of Poisson neurons and wires them to the first excitatory layer of the network. The input layer is technically 3D, but we're treating it as 1d where it list filters for each row,col coord proceeding along each column and then down the rows
    """
    print("########## WIRING INPUT LAYER ##########")
    exc_neuron_layer_1 = exc_neuron_specs.neuron_groups[1]
    num_images, height, width, num_filters = _3d_poisson_rates.shape

    flat_poisson_inputs, hcf = generate_test_train_flat_poisson(
        _3d_poisson_rates,
        no_epochs,
        stimulus_exposure_time,
        stimulus_exposure_time_test,
        no_test_epochs,
    )
    num_images, num_filtered_pixels = flat_poisson_inputs.shape
    num_neurons = exc_neuron_layer_1.N

    timed_input = generate_timed_array_from_flat_poisson_inputs(
        flat_poisson_inputs,
        beta,
        hcf,
    )
    poisson_neurons = PoissonGroup(
        num_filtered_pixels,
        rates="timed_input(t,i)",
        name="p_0",
    )
    network.add(poisson_neurons)
    exc_neuron_specs.add_neurons(0, poisson_neurons)
    synapses = Synapses(
        poisson_neurons,
        exc_neuron_layer_1,
        method="rk4",
        on_pre="""ge += 30 * nS
""",
        name="pfe_post_0",
    ) #ga_post += 6 * nS


    network.add(synapses)
    print(f"GENERATING INPUT DATA")
    # why does this need num_images?
    indices = generate_indices(
        height,
        width,
        num_filters,
        num_neurons,
        radius,
        avg_no_neurons=avg_no_neurons,
    )

    # Flatten indices
    flat_indices = []
    flat_j = []

    for j, pre_indices in enumerate(indices):  # indices is a list of lists
        flat_indices.extend(pre_indices)  # Append all pre-synaptic indices
        flat_j.extend(
            [j] * len(pre_indices)
        )  # Repeat post-synaptic index for each connection

    # Convert to NumPy arrays
    flat_indices = np.array(flat_indices, dtype=int)
    flat_j = np.array(flat_j, dtype=int)
    synapses.connect(i=flat_indices, j=flat_j)
    return timed_input, poisson_neurons


def index_to_wh_coordinates(index, width, filters):
    # Calculate width coordinate
    w = (index // filters) % width
    # Calculate height coordinate
    h = index // (filters * width)
    return (
        h,
        w,
    )


def grab_indices_within_radius(h, w, height, width, filters, radius):
    # Calculate the coordinates of the pixels within the radius (all filters are included)
    indices = []
    for i in range(height):
        for j in range(width):
            if sqrt((h - i) ** 2 + (w - j) ** 2) <= radius:
                for f in range(filters):
                    value = i * width * filters + j * filters + f
                    indices.append(int(value))
    return indices


def generate_indices(
    height,
    width,
    num_filters,
    num_neurons,
    radius,
    avg_no_neurons=1,
):
    indices = [[] for _ in range(num_neurons)]
    indices_len = []
    length_neurons = sqrt(num_neurons)
    for j in range(num_neurons):
        post_h = j // length_neurons
        post_w = j % length_neurons
        scale = height / length_neurons
        h = int(post_h * scale)
        w = int(post_w * scale)
        # print(j)
        indices[j] = grab_indices_within_radius(
            h, w, height, width, num_filters, radius
        )
        indices_len.append(len(indices[j]))
    current_avg_no = np.mean(indices_len)
    connection_prob = avg_no_neurons / current_avg_no
    print("connection_prob: ", connection_prob)
    for j, index_list in enumerate(indices):
        indices[j] = [
            index for index in index_list if np.random.rand() < connection_prob
        ]
    print(f"the final length of input indices is {len(indices)}")
    print(len(indices))
    return indices


def wire_input_layer(
    network: Network, exc_neuron_specs: NeuronSpecs, timed_input, epoch_length
):
    pass

def create_network(
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
    stimulus_exposure_time,
    stimulus_exposure_time_test,
    _3d_poisson_rates,
    no_epochs,
    no_test_epochs,
):
    print("Creating Neuron Groups")
    create_neuron_groups(
        network,
        N_LAYERS,
        exc_neuron_specs,
        inh_neuron_specs)
    
    print("Creating Synapse Groups")
    create_synapse_groups(
        network,
        N_LAYERS,
        RADII,
        AVG_NO_CONNECTIONS,
        exc_neuron_specs,
        inh_neuron_specs,
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    )
    print("Creating inputs and input layer")
    timed_input, poisson_neurons = wire_input_layer_brian(
        network,
        exc_neuron_specs,
        _3d_poisson_rates,
        beta=6000,
        radius=2,
        avg_no_neurons=AVG_NO_CONNECTIONS["efe"][0],
        stimulus_exposure_time=stimulus_exposure_time,
        stimulus_exposure_time_test=stimulus_exposure_time_test,
        no_epochs=no_epochs,
        no_test_epochs=1,
    )

    return timed_input, poisson_neurons

