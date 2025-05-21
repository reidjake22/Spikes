from brian2 import *
from  network.NeuronSpecs import NeuronSpecs
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