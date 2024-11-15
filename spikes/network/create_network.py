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
from .synapses import StdpSynapseSpecs, NonStdpSynapseSpecs


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

    Raises:
    -------
        None

    Example Usage:
    --------------
        TODO
    Notes:
    -------
        We want to add flexibility in specifying the number of neurons and synapses,
    and whether we do back and lateral.
    """

    # Iterate over each layer and create neurons based on their types
    print(f"creating {n_layers} neuron layers")
    for layer in range(1, n_layers + 1):
        print(f"creating layer {layer}")

        # Create excitatory and inhibitory neuron groups for each layer
        exc_neuron_specs.create_neurons(layer, target_network=network)
        inh_neuron_specs.create_neurons(layer, target_network=network)


def create_synapse_groups(
    network,
    n_layers,
    exc_neuron_specs: NeuronSpecs,
    inh_neuron_specs: NeuronSpecs,
    stdp_synapse_specs: StdpSynapseSpecs,
    non_stdp_synapse_specs: NonStdpSynapseSpecs,
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
    stdp_synapse_specs (StdpSynapseSpecs):
        Specifications for the STDP synapse group.
    non_stdp_synapse_specs (NonStdpSynapseSpecs):
        Specifications for the non-STDP synapse group.

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

    # Iterate over each layer and create synapses based on their types
    print(f"creating {n_layers} synapse layers")
    for layer in range(1, n_layers + 1):
        print(f"creating synapses for layer {layer}")
        # Create EE synapses for all layers except the last
        if not layer == n_layers:
            # create E-E synapses
            print(f"creating E-E synapses for layer {layer}")
            stdp_synapse_specs.create_synapses(
                layer,
                exc_neuron_specs,
                exc_neuron_specs,
                radius=2,
                target_network=network,
            )

        # create E-I synapses
        print(f"creating E-I synapses for layer {layer}")
        non_stdp_synapse_specs.create_synapses(
            layer, exc_neuron_specs, inh_neuron_specs, radius=2, target_network=network
        )

        # create I-E synapses
        print(f"creating I-E synapses for layer {layer}")
        non_stdp_synapse_specs.create_synapses(
            layer, inh_neuron_specs, exc_neuron_specs, radius=2, target_network=network
        )

        # create I-I synapses
        print(f"creating I-I synapses for layer {layer}")
        non_stdp_synapse_specs.create_synapses(
            layer, inh_neuron_specs, inh_neuron_specs, radius=2, target_network=network
        )


def wire_input_layer(
    network: Network, exc_neuron_specs: NeuronSpecs, timed_input, epoch_length
):
    """
    Overview:
    --------
    Creates an input layer of Poisson neurons and wires them to the first
    excitatory layer of the network.

    Parameters:
    -----------
    network : Network
        The neural network to which the input layer and synapses will be added.
    exc_neuron_specs : NeuronSpecs
        Specifications of the excitatory neurons, including neuron groups.
    timed_input : TimedArray
        Defines the time-dependent input rates for the Poisson neurons.

    Returns:
    --------
        input_synapses (Synapses):
        The synapses created between the Poisson input layer and the first excitatory layer.

    Raises:
    -------
        None

    Example Usage:
    --------------

    Notes:
    ------

    """
    exc_neuron_layer_1 = exc_neuron_specs.neuron_groups["excitatory_layer_1"]
    timed_input = timed_input
    poisson_neurons = PoissonGroup(
        exc_neuron_layer_1.N,
        rates="timed_input((t%epoch_length),i)",  # let's the timed_input indefinitely loop I hope
        name="poisson_layer_0",
    )
    network.add(poisson_neurons)
    input_synapses = Synapses(
        poisson_neurons,
        exc_neuron_layer_1,
        on_pre="ge += input_lambda_e",
        name="poisson_excitatory_0",
    )
    network.add(input_synapses)
    input_synapses.connect(i="j")
    return input_synapses


# Function to retrieve an object by name without running
# def get_object_by_name(network, name):
#     """
#     Search for an object by its name in a given Network.

#     Parameters:
#     - network: The Network object to search in.
#     - name: The name of the object to find (string).

#     Returns:
#     - The object if found, or None if not found.
#     """

#     for obj in network.objects:
#         print(obj[0][0])
#         # if hasattr(obj.NeuronGroup, "name"):
#         if obj.name == name:
#             return obj
#     raise Warning(f"Object with name '{name}' not found in the network.")


# def generate_inputs(input: np.array):
#     """
#     input vector is a 3D array. We're assuming it goes index, height, width
#     """
#     stimulus_exposure_time = 10 * ms
#     index, height, width = input.shape
#     print(
#         f"the number of images for the input is {index}, the height is {height}, and the width is {width}"
#     )

#     # Gonna have to sort the dimensions here unfortunately - what a hassle
#     collapsed_input = np.array([input.flatten() for image in input])
#     collapsed_input_hz = collapsed_input * 100 * hertz
#     timed_input = TimedArray(collapsed_input_hz, dt=stimulus_exposure_time)
#     poisson_neurons = PoissonGroup(
#         N=int(height * width),
#         rates="timed_input(t,i)",
#         name="input_layer_0",
#         # namespace={"timed_input": timed_input},
#     )
#     return poisson_neurons, timed_input


# def connect_to_inputs(
#     network: Network,
#     exc_neuron_specs: NeuronSpecs,
#     poisson_neurons: NeuronGroup,
#     input_synapse_specs: InputSynapseSpecs,
# ):
#     """
#     Connects the input layer to the first excitatory layer
#     """
#     excitatory_layer_1 = exc_neuron_specs.neuron_groups["excitatory_layer_1"]
#     print(excitatory_layer_1.ge)
#     input_synapse_specs.create_synapses(poisson_neurons, excitatory_layer_1, network)
