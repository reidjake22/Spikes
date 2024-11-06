from brian2 import *
from input import *
from network import *

exc_neuron_specs = NeuronSpecs(
    neuron_type="excitatory",
    length=28,
    cm=500 * pF,
    g_leak=25 * nS,
    v_threshold=-53 * mV,
    v_reset=-57 * mV,
    v_rest=-74 * mV,
    v_reversal_e=0 * mV,
    v_reversal_i=-70 * mV,
    sigma=0.015 * mV,
    t_refract=2 * ms,  # NEED TO ADD THIS
    tau_m=20 * ms,
    tau_ee=2 * ms,
    tau_ie=5 * ms,
)

inh_neuron_specs = NeuronSpecs(
    neuron_type="inhibitory",
    length=14,
    cm=214 * pF,
    g_leak=18 * nS,
    v_threshold=-53 * mV,
    v_reset=-58 * mV,
    v_rest=-82 * mV,
    v_reversal_e=0 * mV,
    v_reversal_i=-70 * mV,
    sigma=0.015 * mV,
    tau_m=12 * ms,
    tau_ei=2 * ms,
    tau_ii=5 * ms,
)
stdp_synapse_specs = StdpSynapseSpecs(
    lambda_e=0.1,
    A_minus=0.1,
    A_plus=0.1,
    alpha_C=0.5,
    alpha_D=0.5,
    tau_c=3 * ms,
    tau_d=5 * ms,
)

layer = 1
exc_neuron_specs.create_neurons(layer)
inh_neuron_specs.create_neurons(layer)

stdp_synapse_specs.create_synapses(exc_neuron_specs, inh_neuron_specs, 2, layer)
stdp_synapse_specs.animate_plots(exc_neuron_specs, inh_neuron_specs, 2, layer)
