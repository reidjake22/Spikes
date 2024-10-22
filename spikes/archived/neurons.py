from brian2 import *

# Equations for simple LIF neuron model used for testing basic structure
eqs = """
    dv/dt = (I - v) / (10*ms) : 1  # Simple leaky integrate-and-fire model
    I : 1  # Input current
    x : 1  # Neuron x-coordinate
    y : 1  # Neuron y-coordinate
    """


class NeuronSpecs:
    def __init__(self, neuron_type="excitatory", params=None):
        """
        Initializes the neuron model based on the provided neuron type (excitatory, inhibitory, or input).

        Args:
        neuron_type: Type of neuron model ("excitatory", "inhibitory", or "input"). Default is "excitatory".
        params: Dictionary of custom neuron parameters. If None, default parameters are used.
        """
        # List of equations for different neuron types
        equations_list = {
            "excitatory": """
            dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m*g_leak) + sigma*xi*(tau_m)**-0.5 : volt
            dge/dt = -ge/tau_ee : siemens
            dgi/dt = -gi/tau_ie : siemens
            Cm : farad  # Membrane capacitance
            g_leak : siemens  # Leak conductance
            V_rest : volt  # Resting potential
            V_reversal_e : volt  # Reversal potential for excitatory synapses
            V_reversal_i : volt  # Reversal potential for inhibitory synapses
            sigma : volt  # Noise term
            tau_m : second  # Membrane time constant
            tau_ee : second  # Time constant for excitatory-excitatory synapses
            tau_ie : second  # Time constant for inhibitory-excitatory synapses
            x : integer (constant)
            y : integer (constant)
            """,
            "inhibitory": """
            dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m * g_leak) + sigma*xi*(tau_m)**-0.5 : volt
            dge/dt = -ge/tau_ei : siemens
            dgi/dt = -gi/tau_ii : siemens
            Cm : farad
            g_leak : siemens
            V_rest : volt
            V_reversal_e : volt
            V_reversal_i : volt
            sigma : volt
            tau_m : second
            tau_ei : second
            tau_ii : second
            """,
            "input": """
            dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m*g_leak) + sigma*xi*(tau_m)**-0.5 : volt
            dge/dt = -ge/tau_ee : siemens
            dgi/dt = -gi/tau_ie : siemens
            Cm : farad
            g_leak : siemens
            V_rest : volt
            V_reversal_e : volt
            V_reversal_i : volt
            sigma : volt
            tau_m : second
            tau_ee : second
            tau_ie : second
            x : integer (constant)
            y : integer (constant)
            """,
        }

        # Default parameters for excitatory neurons
        default_params_excitatory = {
            "Cm": 500 * pF,  # Membrane capacitance
            "g_leak": 25 * nS,  # Leak conductance
            "V_rest": -74 * mV,  # Resting membrane potential
            "V_threshold": -53 * mV,  # Firing threshold
            "V_reset": -57 * mV,  # Reset potential after spike
            "V_reversal_e": 0 * mV,  # Excitatory reversal potential
            "V_reversal_i": -70 * mV,  # Inhibitory reversal potential
            "t_refract": 2 * ms,  # Refractory period after spike
            "sigma": 0.015 * mV,  # Noise intensity
            "tau_m": 20 * ms,  # Membrane time constant
            "tau_ee": 2 * ms,  # Synaptic time constant for excitatory synapses
            "tau_ie": 2
            * ms,  # Synaptic time constant for inhibitory synapses onto excitatory neurons
        }

        # Default parameters for inhibitory neurons
        default_params_inhibitory = {
            "Cm": 214 * pF,  # Membrane capacitance
            "g_leak": 28 * nS,  # Leak conductance
            "V_rest": -82 * mV,  # Resting membrane potential
            "V_threshold": -53 * mV,  # Firing threshold
            "V_reset": -58 * mV,  # Reset potential after spike
            "V_reversal_e": 0 * mV,  # Excitatory reversal potential
            "V_reversal_i": -70 * mV,  # Inhibitory reversal potential
            "t_refract": 2 * ms,  # Refractory period
            "sigma": 0.015 * mV,  # Noise intensity
            "tau_m": 12 * ms,  # Membrane time constant
            "tau_ei": 5
            * ms,  # Synaptic time constant for excitatory synapses onto inhibitory neurons
            "tau_ii": 5
            * ms,  # Synaptic time constant for inhibitory synapses onto inhibitory neurons
        }

        self.neuron_type = neuron_type  # Store the neuron type

        # Select default parameters based on neuron type
        if self.neuron_type == "excitatory" or self.neuron_type == "input":
            default_params = default_params_excitatory
        else:
            default_params = default_params_inhibitory

        # Update default parameters with user-provided custom parameters (if any)
        if params is not None:
            self.params = {
                **default_params,
                **params,
            }  # Custom parameters override defaults
        else:
            self.params = default_params

        print(self.params)

        # Select the appropriate equation for the neuron type
        self.equations = equations_list[neuron_type]

    def create_neuron_layer(self, size):
        """
        Creates a NeuronGroup of specified size using the model parameters.

        Args:
        size: Dimension size for the neuron group (size x size neurons)

        Returns:
        NeuronGroup object containing the neurons
        """
        # Create the NeuronGroup using the selected equations and parameters
        neurons = NeuronGroup(
            N=int(size * size),  # Total number of neurons (size x size)
            model=self.equations,  # Neuron equations
            threshold="v > V_threshold",  # Spiking threshold condition
            reset="v = V_reset",  # Reset condition after spike
            refractory=self.params["t_refract"],  # Refractory period
            method="linear",  # Numerical integration method
        )

        # Assign parameter values to the NeuronGroup
        neurons.Cm = self.params["Cm"]
        neurons.g_leak = self.params["g_leak"]
        neurons.V_rest = self.params["V_rest"]
        neurons.sigma = self.params["sigma"]
        neurons.ge = 0 * siemens  # Initial excitatory synaptic conductance
        neurons.gi = 0 * siemens  # Initial inhibitory synaptic conductance

        # Add x, y coordinates as attributes to the neuron group using add_attribute
        neuron_indices = [(i, j) for i in range(size) for j in range(size)]
        neurons.x = [i for i, j in neuron_indices]  # x-coordinate
        neurons.y = [j for i, j in neuron_indices]  # y-coordinate

        # Set synaptic time constants based on neuron type
        if self.neuron_type == "excitatory" or self.neuron_type == "input":
            neurons.tau_ee = self.params[
                "tau_ee"
            ]  # Excitatory-excitatory time constant
            neurons.tau_ie = self.params[
                "tau_ie"
            ]  # Inhibitory-excitatory time constant
        elif self.neuron_type == "inhibitory":
            neurons.tau_ei = self.params[
                "tau_ei"
            ]  # Excitatory-inhibitory time constant
            neurons.tau_ii = self.params[
                "tau_ii"
            ]  # Inhibitory-inhibitory time constant

        return neurons
