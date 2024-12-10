# network/__init__.py

# Import from equations.py
from .visualisation import (
    visualise_poisson_inputs,
    plot_synapse_distributions,
    three_dim_visualise_synapses,
    display_input_activity,
)
from .analysis import describe_network_components

# Specify the items to expose in * imports
__all__ = [
    "visualise_poisson_inputs",
    "plot_synapse_distributions",
    "three_dim_visualise_synapses",
    "describe_network_components",
    "display_input_activity",
]
