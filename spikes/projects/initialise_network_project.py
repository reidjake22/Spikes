# This takes network settings and initialises it within a project directory for subsequent use
# This means it 
from brian2 import *
import os
from network import *
from input import *
from projects import set_project_environment
import numpy as np
import matplotlib.pyplot as plt
 
def initialise_network_project(
    project_name: str,
    base_dir: str,
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
):
    project_base = set_project_environment(base_dir, project_name)
    print("Creating Neuron Groups")
    create_neuron_groups(
        network,
        N_LAYERS,
        exc_neuron_specs,
        inh_neuron_specs)
    
    print("Creating Synapse Groups")
    configs_path = os.path.join(base_dir, project_name, "configs")
    network_configs_path = os.path.join(configs_path, "network")
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
        storage="store",
        storage_path=network_configs_path, 
           )
    print("Creating Input stuff")
    # create gabor filters
    input_configs_path = os.path.join(configs_path, "input")
    gabor_save_path = os.path.join(input_configs_path, "filters")
    print(f"Creating gabor filters at {gabor_save_path}")
    lambdas = [0.8]  # Wavelengths
    betas = [1]  # Scaling factor for bandwidth
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Orientations
    psis = [0, np.pi]  # Phase offsets
    gammas = [0.5]  # Aspect ratio
    size = 6  # Gabor filter sizes

    generate_gabor_filters(
        gabor_save_path,
        lambdas,
        betas,
        thetas,
        psis,
        gammas,
        size,
    )
    # Creates a mapping of the input layer to the first layer of the network
    num_excitatory_neurons = exc_neuron_specs.length ** 2
    print(f"input_configs_path, {input_configs_path}")
    create_receptive_field_mapping(
        num_filters = 8,
        height = 128,
        width = 128,
        grid_shape = (64, 64),
        radius = 8,
        avg_fanin = 50,
        mapping_save_path = input_configs_path,
        rng=np.random.default_rng(42),
    )