from brian2 import *
from .NeuronSpecs import NeuronSpecs
from .SynapseSpecs import SynapseSpecs

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
    storage: str = "none",
    storage_path: str = "initial_config",
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
                storage=storage,
                storage_path=storage_path,
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
            storage=storage,
            storage_path=storage_path,
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
                storage=storage,
                storage_path=storage_path,
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
            storage=storage,
            storage_path=storage_path,
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
            storage=storage,
            storage_path=storage_path,
        )
