# network/__init__.py

# Import from equations.py
from .equations import EquationsContainer

# Import from neurons.py
from .neurons import NeuronSpecs

# Import from synapses.py
<<<<<<< HEAD
from .synapses import SynapseSpecs
=======
from .synapses import (SynapseSpecs,
                       SynapseSpecsInfo)
>>>>>>> jakes_working_repo

# Import from create_network.py
from .create_network import (
    create_neuron_groups,
    create_synapse_groups,
    wire_input_layer,
    wire_input_layer_brian,
<<<<<<< HEAD
=======
    create_network
>>>>>>> jakes_working_repo
)

# Specify the items to expose in * imports
__all__ = [
    "EquationsContainer",
    "NeuronSpecs",
    "SynapseSpecs",
<<<<<<< HEAD
=======
    "SynapseSpecsInfo",
>>>>>>> jakes_working_repo
    "wire_input_layer",
    "wire_input_layer_brian",
    "create_neuron_groups",
    "create_synapse_groups",
    "create_network",
]
