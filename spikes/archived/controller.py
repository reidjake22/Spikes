from brian2 import *


network = Network()
n_layers = 3  # Number of layers to create

# Currently length is stores as a parameter in the NeuronSpecs class
# This is less than optimal, but it works for now
# Define neuron specifications for excitatory neurons
exc_neuron_specs = NeuronSpecs(
    neuron_type="excitatory",
    length=8,
)

# Define neuron specifications for inhibitory neuron
inh_neuron_specs = NeuronSpecs(
    neuron_type="inhibitory",
    length=4,
)

# Define neuron specifications for input neurons
input_neuron_specs = NeuronSpecs(
    neuron_type="input",
    length=8,
)

# Define STDP synapse specifications
stdp_synapse_specs = StdpSynapseSpecs(
    lambda_e=0.1,
    A_minus=0.1,
    A_plus=0.1,
    alpha_C=0.1,
    alpha_D=0.1,
    tau_pre=0.1,
    tau_post=0.1,
)

# Define non-STDP synapse specifications
non_stdp_synapse_specs = NonStdpSynapseSpecs(
    lambda_e=0.1,
    lambda_i=0.1,
)

# Define input synapse specifications
input_synapse_specs = InputSynapseSpecs(
    lambda_e=0.1,
)

# Create Synapses
create_neuron_groups(
    network, n_layers, exc_neuron_specs, inh_neuron_specs, input_neuron_specs
)

# Create Synapses
create_synapse_groups(
    network,
    n_layers,
    stdp_synapse_specs,
    non_stdp_synapse_specs,
)
inputs = np.array([0])
poisson_neurons = generate_inputs(inputs)
connect_to_inputs(network, poisson_neurons, non_stdp_synapse_specs)
