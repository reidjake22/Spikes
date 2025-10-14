import sys
import os
sys.path.insert(0, r"/home/jake/Document/Spikes/spikes")
    
if __name__ == "__main__":
    print("Imports")
    from brian2 import *
    from network import *
    from input import *
    from projects import *
    import numpy as np
    import matplotlib.pyplot as plt
    equations_container = EquationsContainer()
    network = Network()
    project = "mnist_class_wip"
    dir = os.path.dirname(os.path.abspath(__file__))
    # Set up parameters
    N_layers = 4
    STIMULUS_LENGTH = 100 * ms
    image_height, image_width, num_filters = 128, 128, 8
    grid_size = 64  # For excitatory layers
    RADII = {
        "efe": {1: 8, 2: 12, 3: 16},
        "ele": {1: 2, 2: 2, 3: 2, 4: 2},
        "ebe": {2: 8, 3: 8, 4: 8},
        "eli": {1: 2, 2: 2, 3: 2, 4: 2},
        "ile": {1: 4, 2: 4, 3: 4, 4: 4},
    }

    AVG_NO_CONNECTIONS = {
        "efe": {0: 50, 1: 100, 2: 100, 3: 100},
        "ele": {1: 10, 2: 10, 3: 10, 4: 10},
        "ebe": {1: 10, 2: 10, 3: 10, 4: 10},
        "eli": {1: 10, 2: 10, 3: 10, 4: 10},
        "ile": {1: 30, 2: 30, 3: 30, 4: 30},
    }

    print("Defining Excitatory Neurons")
    exc_neuron_specs = NeuronSpecs(
        neuron_type="e",
        length=64,
        cm=500 * pF,
        g_leak=25 * nS,
        v_threshold=-53 * mV,
        v_reset=-57 * mV,
        v_rest=-74 * mV,
        v_reversal_e=0 * mV,
        v_reversal_i=-70 * mV,
        v_reversal_a=-90 * mV,
        sigma=0.015 * mV,
        t_refract=2 * ms,
        tau_m=20 * ms,
        tau_ee=2 * ms,
        tau_ie=5 * ms,
        tau_a=80 * ms,
    )
    print("Defining Inhibitory Neurons")
    inh_neuron_specs = NeuronSpecs(
        neuron_type="i",
        length=32,
        cm=214 * pF,
        g_leak=18 * nS,
        v_threshold=-53 * mV,
        v_reset=-58 * mV,
        v_rest=-82 * mV,
        v_reversal_e=0 * mV,
        v_reversal_i=-70 * mV,
        sigma=0.015 * mV,
        t_refract=2 * ms,
        tau_m=12 * ms,
        tau_ei=2 * ms,
        tau_ii=5 * ms,
    )
    print("Defining Synapse Specifications")
    print("Defining EFE")
    efe_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="f",
        name="efe",
        lambda_e=30 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    print("Defining ELE")
    ele_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="l",
        name="ele",
        lambda_e=20 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    print("Defining EBE")
    ebe_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["stdp_model"],
        on_pre=equations_container.synaptic_equations["stdp_on_pre"],
        on_post=equations_container.synaptic_equations["stdp_on_post"],
        type="b",
        name="ebe",
        lambda_e=20 * nS,
        alpha_C=0.5,
        alpha_D=0.5,
        tau_c=5 * ms,
        tau_d=5 * ms,
        learning_rate=0.04,
    )
    print("Defining ELI")
    eli_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["excit_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["excit_non_stdp_on_pre"],
        type="l",
        name="eli",
        lambda_e=20 * nS,
    )
    print("Defining ILE")
    ile_synapse_specs = SynapseSpecs(
        model=equations_container.synaptic_equations["inhib_non_stdp_model"],
        on_pre=equations_container.synaptic_equations["inhib_non_stdp_on_pre"],
        type="l",
        name="ile",
        lambda_i=30 * nS,
    )
    print("Creating Network")
    
    initialise_network_project(
        "mnist_class_wip",
        dir,
        network,
        4,
        exc_neuron_specs,
        inh_neuron_specs,
        RADII,
        AVG_NO_CONNECTIONS,
        efe_synapse_specs,
        ele_synapse_specs,
        ebe_synapse_specs,
        eli_synapse_specs,
        ile_synapse_specs,
    )