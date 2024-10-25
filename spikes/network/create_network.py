# Add the directory containing the modules to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the necessary modules
from brian2 import *
from neurons import NeuronSpecs
from spikes.network.synapses import StdpSynapseSpecs, NonStdpSynapseSpecs
import os
import sys


def create_neuron_groups(
    network,
    n_layers,
    input_neuron_specs: NeuronSpecs,
    exc_neuron_specs: NeuronSpecs,
    inh_neuron_specs: NeuronSpecs,
):
    """
    Creates neuron groups for each layer, including input, excitatory, and inhibitory neurons.

    Parameters:
    -----------
    n_layers : int
        Number of layers in the network.
    input_neuron_specs : NeuronSpecs
        Specifications for the input neuron group.
    exc_neuron_specs : NeuronSpecs
        Specifications for the excitatory neuron group.
    inh_neuron_specs : NeuronSpecs
        Specifications for the inhibitory neuron group.
    """

    # Iterate over each layer and create neurons based on their types
    print(f"creating {n_layers} neuron layers")
    for layer in n_layers:
        print(f"creating layer {layer + 1}")
        # Create excitatory and inhibitory neuron groups for each layer
        exc_neuron_specs.create_neurons(layer, target=network)
        inh_neuron_specs.create_neurons(layer, target=network)


def create_synapse_groups(
    network,
    n_layers,
    stdp_synapse_specs: StdpSynapseSpecs,
    non_stdp_synapse_specs: NonStdpSynapseSpecs,
):
    """
    Creates synapse groups for each layer, including STDP and non-STDP synapses.

    Parameters:
    -----------
    n_layers : int
        Number of layers in the network.
    stdp_synapse_specs : StdpSynapseSpecs
        Specifications for the STDP synapse group.
    non_stdp_synapse_specs : NonStdpSynapseSpecs
        Specifications for the non-STDP synapse group.
    """

    # Iterate over each layer and create synapses based on their types

    connect_to_inputs()
    for layer in range(n_layers):
        if not layer == n_layers:
            # create E-E synapses
            afferent_group = get_object_by_name(network, f"excitatory_layer_{layer}")
            efferent_group = get_object_by_name(network, f"excitatory_layer_{layer+1}")
            stdp_synapse_specs.construct(layer, afferent_group, efferent_group, network)

        # create E-I synapses
        afferent_group = get_object_by_name(network, f"excitatory_layer_{layer}")
        efferent_group = get_object_by_name(network, f"inhibitory_layer_{layer}")
        non_stdp_synapse_specs.construct(layer, afferent_group, efferent_group, network)

        # create I-E synapses
        afferent_group = get_object_by_name(network, f"inhibitory_layer_{layer}")
        efferent_group = get_object_by_name(network, f"excitatory_layer_{layer}")
        non_stdp_synapse_specs.construct(layer, afferent_group, efferent_group, network)

        # create I-I synapses
        afferent_group = get_object_by_name(network, f"inhibitory_layer_{layer}")
        efferent_group = get_object_by_name(network, f"inhibitory_layer_{layer}")
        non_stdp_synapse_specs.construct(layer, afferent_group, efferent_group, network)


# Function to retrieve an object by name without running
def get_object_by_name(network, name):
    """
    Search for an object by its name in a given Network.

    Parameters:
    - network: The Network object to search in.
    - name: The name of the object to find (string).

    Returns:
    - The object if found, or None if not found.
    """

    for obj in network.objects:
        if hasattr(obj, "name") and obj.name == name:
            return obj
    raise Warning(f"Object with name '{name}' not found in the network.")


def generate_inputs(input: np.array):
    """
    input vector is a 3D array
    """
    stimulus_exposure_time = 10 * ms
    x, y, z = array.shape

    # Gonna have to sort the dimensions here unfortunately - what a hassle
    collapsed_input = input.reshape(x * y, z)
    timed_input = TimedArray(collapsed_input, dt=stimulus_exposure_time)
    poisson_neurons = PoissonGroup(
        N=x * y, rates="timed_input(t,i)", name="input_layer_0"
    )
    return poisson_neurons


def connect_to_inputs(
    network, poisson_neurons, non_stdp_synapse_specs: NonStdpSynapseSpecs
):
    """
    Connects the input layer to the first excitatory layer
    """
    excitatory_layer_1 = get_object_by_name(network, "excitatory_layer_1")
    non_stdp_synapse_specs.construct(0, poisson_neurons, excitatory_layer_1, network)
