"""
This function creates a network of neurons and synapses based on the provided specifications.
It initializes neuron groups, synapse groups, and an input layer. The function also allows for saving or loading the network configuration.
The poisson input neurons are 1:1.
"""
from brian2 import *
from .create_neuron_groups import create_neuron_groups
from .create_synapse_groups import create_synapse_groups
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
    storage: str = "none",  # Options: "save", "load", "none"
    storage_path: str = "/configs/initial_config",
):
    print(f"Creating Neuron Groups (storage mode: {storage}, path: {storage_path})")
    create_neuron_groups(
        network,
        N_LAYERS,
        exc_neuron_specs,
        inh_neuron_specs,)
    
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
        storage,
        storage_path,
    )
    print("Creating input layer")
    # Okay so now we're gonna create a new object poisson_rates that has a 3d array which already took into account a mapping. This mapping and its rates are saved somewhere.
    num_excitatory_neurons = exc_neuron_specs.neuron_groups[1].N
    print(f"Number of excitatory neurons: {num_excitatory_neurons}")
    timed_array = TimedArray(
        1000 * Hz * zeros((num_excitatory_neurons,)),
        dt=250*ms,
        name="input_rates",
    )
    
    poisson_neurons = NeuronGroup(num_excitatory_neurons, 'rates = timed_array(t,i) : Hz', threshold='rand()<rates*dt', name="p_0")
    exc_neuron_specs.add_neurons(0, poisson_neurons)
    network.add(poisson_neurons)
    synapses = Synapses(
        poisson_neurons,
        exc_neuron_specs.neuron_groups[1],
        method="rk4",
        on_pre="""ge += 30 * nS
""",
        name="pfe_post_0",
    )
    network.add(synapses)
    synapses.connect(i="j")
