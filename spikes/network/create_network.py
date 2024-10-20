"""
basically we're gonna have to create a few functions that can cleanly handle 
building the network and running the network
The idea would be that swapping and changing networks should be easy so that
the whole process is quite flexible (suppose you occassionally wanted to do several runs,
or maybe try a different network configuration etc)
Maybe a Class with default values for neurons, but the option to change them, with the ultimate ability to pass it to the network during instantiation
This always seems to work out awkward though!
It might genuinely just be a question of passing a dictionary, with some check that it's the right shape
OOOh
okay so we have several networks, at the top of each network we run the function - "check parameters"
we want a:
"generate neurons function"
"generate STDP synapses function"
"connect synapses function"
"generate network function"
want parts of the generation process modular, but defo 
e want the result to be all the objects we make included in the network we can pass
"""

from brian2 import *


class EquationsContainer:
    """
    A container to store different sets of equations for neurons, synapses,
    and other models in the network.

    Attributes:
    -----------
    neuron_equations : dict
        Dictionary storing equations for different types of neurons (e.g., excitatory, inhibitory).
    synaptic_equations : dict
        Dictionary storing equations for different types of synapses (future expansion).
    other_equations : dict
        Dictionary storing other types of equations if needed.
    """

    def __init__(self):
        """Initialize empty dictionaries for neuron, synaptic, and other equations."""
        # Initialiase the dictionaries for different equations
        self.neuron_equations = {}
        self.synaptic_equations = {}
        self.other_equations = {}

        # Call function to add basic equations
        self.add_basic_equations()

    def add_basic_equations(self):
        """
        Adds basic neuron equations for excitatory, inhibitory, and input neurons
        to the neuron_equations dictionary.
        """

        # Define excitatory neuron equations
        excitatory_model = Equations(
            """
            dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m*g_leak) + sigma*xi*(tau_m)**-0.5 : volt
            dge/dt = -ge/tau_ee : siemens
            dgi/dt = -gi/tau_ie : siemens
            Cm : farad  # Membrane capacitance
            g_leak : siemens  # Leak conductance
            V_rest : volt  # Resting potential
            V_reversal_e : volt  # Reversal potential for excitatory synapses
            V_reversal_i : volt  # Reversal potential for inhibitory synapses
            't_refract': second # Refractory period
            sigma : volt  # Noise term
            tau_m : second  # Membrane time constant
            tau_ee : second  # Time constant for excitatory-excitatory synapses
            tau_ie : second  # Time constant for inhibitory-excitatory synapses
            x : integer (constant)
            y : integer (constant)
            """
        )

        # Define inhibitory neuron equations
        inhibitory_model = Equations(
            """
            dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m * g_leak) + sigma*xi*(tau_m)**-0.5 : volt
            dge/dt = -ge/tau_ei : siemens
            dgi/dt = -gi/tau_ii : siemens
            Cm : farad
            g_leak : siemens
            V_rest : volt
            V_reversal_e : volt
            V_reversal_i : volt
            t_refract: second # Refractory period
            sigma : volt
            tau_m : second
            tau_ei : second
            tau_ii : second
            """
        )

        # Define input neuron equations
        input_model = Equations(
            """
            dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m*g_leak) + sigma*xi*(tau_m)**-0.5 : volt
            dge/dt = -ge/tau_ee : siemens
            dgi/dt = -gi/tau_ie : siemens
            Cm : farad
            g_leak : siemens
            V_rest : volt
            V_reversal_e : volt
            V_reversal_i : volt
            t_refract: second # Refractory period
            sigma : volt
            tau_m : second
            tau_ee : second
            tau_ie : second
            x : integer (constant)
            y : integer (constant)
            """
        )

        # Define input neuron equations
        self.neuron_equations["excitatory"] = excitatory_model
        self.neuron_equations["inhibitory"] = inhibitory_model
        self.neuron_equations["input"] = input_model

    def add_equation(self, eq_type, equation_name, equation_body):
        """
        Adds a custom equation to the appropriate dictionary based on the type.

        Parameters:
        -----------
        eq_type : str
            The type of equation ('neuron', 'synaptic', or 'other').
        equation_name : str
            Name of the equation to be stored.
        equation_body : Equations
            The body of the equation as a Brian2 Equations object.

        Raises:
        -------
        ValueError if an unknown equation type is provided.
        """

        # Depending on the equation type, store it in the appropriate dictionary
        if eq_type == "neuron":
            self.neuron_equations[equation_name] = equation_body
        elif eq_type == "synaptic":
            self.synaptic_equations[equation_name] = equation_body
        elif eq_type == "other":
            self.other_equations[equation_name] = equation_body
        else:
            raise ValueError(f"Unknown equation type: {eq_type}")


# Initialize the global equations container
equations = EquationsContainer()


class NeuronParameters:
    """
    Class to hold and validate neuron parameters.

    Parameters:
    -----------
    cm : farad | Membrane capacitance.
    g_leak : siemens
        Leak conductance.
    v_leak : volt
        Leak potential.
    v_threshold : volt
        Firing threshold voltage.
    v_reset : volt
        Reset voltage after a spike.
    v_rest : volt
        Resting potential.
    v_reversal_e : volt
        Reversal potential for excitatory synapses.
    v_reversal_i : volt
        Reversal potential for inhibitory synapses.
    sigma : volt
        Noise term.
    tau_m : second
        Membrane time constant.
    tau_ee : second
        Time constant for excitatory-excitatory synapses.
    tau_ei : second
        Time constant for excitatory-inhibitory synapses.
    tau_ie : second
        Time constant for inhibitory-excitatory synapses.
    tau_ii : second
        Time constant for inhibitory-inhibitory synapses.
    neuron_type : str
        Type of neuron ('excitatory' or 'inhibitory').
    """

    def __init__(
        self,
        cm,
        g_leak,
        v_leak,
        v_threshold,
        v_reset,
        v_rest,
        v_reversal_e,
        v_reversal_i,
        sigma,
        tau_m,
        tau_ee,
        tau_ei,
        tau_ie,
        tau_ii,
        neuron_type,
    ):
        # Validate the parameters provided by calling the check_valid_parameters function
        self.check_valid_parameters(
            cm=cm,
            g_leak=g_leak,
            v_leak=v_leak,
            v_threshold=v_threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            v_reversal_e=v_reversal_e,
            v_reversal_i=v_reversal_i,
            sigma=sigma,
            tau_m=tau_m,
            tau_ee=tau_ee,
            tau_ei=tau_ei,
            tau_ie=tau_ie,
            tau_ii=tau_ii,
            neuron_type=neuron_type,
        )

        # Assign validated parameters to the class attributes
        self.cm = cm
        self.g_leak = g_leak
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.v_reversal_e = v_reversal_e
        self.v_reversal_i = v_reversal_i
        self.sigma = sigma
        self.tau_m = tau_m
        self.tau_ee = tau_ee
        self.tau_ei = tau_ei
        self.tau_ie = tau_ie
        self.tau_ii = tau_ii

    def check_valid_parameters(self, **params):
        """
        Checks if all provided parameters are valid (non-None).
        Also warns if the neuron_type is unspecified but tau values imply a specific type.

        Parameters:
        -----------
        params : dict
            Dictionary of all neuron parameters to be validated.

        Raises:
        -------
        ValueError if any required parameter is not specified.
        """
        # Explicitly remove "neuron_type" from the params dictionary before checking
        neuron_type = params.pop("neuron_type", None)  # Remove and store neuron_type

        # Ensure all parameters are provided (i.e., not None)
        for name, param in params.items():
            if param is None:
                raise ValueError(f"Parameter {name} must be specified (non-None).")

        # Warn or check based on tau values if neuron_type is unspecified or mismatched
        if neuron_type is None:
            if params["tau_ee"] and params["tau_ie"]:
                raise Warning(
                    f"Neuron type unspecified. It appears to be excitatory based on tau_ee ({params['tau_ee']}) and tau_ie ({params['tau_ie']}). "
                    f"Consider specifying neuron_type as 'excitatory' for clarity."
                )
            elif params["tau_ei"] and params["tau_ii"]:
                raise Warning(
                    f"Neuron type unspecified. It appears to be inhibitory based on tau_ei ({params['tau_ei']}) and tau_ii ({params['tau_ii']}). "
                    f"Consider specifying neuron_type as 'inhibitory' for clarity."
                )

        # Explicitly check if excitatory parameters are consistent
        elif neuron_type == "excitatory":
            if not (
                params["tau_ee"]
                and params["tau_ie"]
                and not (params["tau_ii"] or params["tau_ei"])
            ):
                raise ValueError(
                    f"Mismatch in conductance parameters for type 'excitatory'. Check the following:\n"
                    f"tau_ee: {params['tau_ee']} [expected positive value]\n"
                    f"tau_ei: {params['tau_ei']} [expected no value]\n"
                    f"tau_ie: {params['tau_ie']} [expected positive value]\n"
                    f"tau_ii: {params['tau_ii']} [expected no value]"
                )

        # Explicitly check if inhibitory parameters are consistent
        elif neuron_type == "inhibitory":
            if not (
                params["tau_ei"]
                and params["tau_ii"]
                and not (params["tau_ie"] or params["tau_ee"])
            ):
                raise ValueError(
                    f"Mismatch in conductance parameters for type 'inhibitory'. Check the following:\n"
                    f"tau_ee: {params['tau_ee']} [expected no value]\n"
                    f"tau_ei: {params['tau_ei']} [expected positive value]\n"
                    f"tau_ie: {params['tau_ie']} [expected no value]\n"
                    f"tau_ii: {params['tau_ii']} [expected positive value]"
                )


class NeuronSpecs:
    """
    Class to hold neuron specifications such as type, size, and corresponding parameters.

    Parameters:
    -----------
    neuron_type : str
        Type of the neuron ('excitatory', 'inhibitory', etc.).
    size : int
        Size of the neuron group (e.g., grid dimensions for spatially organized groups).
    cm, g_leak, v_leak, v_threshold, etc. : various types
        Neuron parameters required to initialize the group.
    """

    def __init__(
        self,
        neuron_type,
        size,
        cm=None,
        g_leak=None,
        v_leak=None,
        v_threshold=None,
        v_reset=None,
        v_rest=None,
        v_reversal_e=None,
        v_reversal_i=None,
        sigma=None,
        tau_m=None,
        tau_ee=None,
        tau_ei=None,
        tau_ie=None,
        tau_ii=None,
    ):
        # Store neuron type and size, and validate neuron parameters
        self.neuron_type = neuron_type
        self.size = size
        self.parameters = NeuronParameters(
            cm,
            g_leak,
            v_leak,
            v_threshold,
            v_reset,
            v_rest,
            v_reversal_e,
            v_reversal_i,
            sigma,
            tau_m,
            tau_ee,
            tau_ei,
            tau_ie,
            tau_ii,
            neuron_type,
        )

    def create_neurons(self, layer, target=None):
        """
        Creates neurons in the given layer.

        Parameters:
        -----------
        layer : int
            The current layer number where neurons should be created.
        target : optional
            The target network to which the neurons should be added (optional).

        Returns:
        --------
        neurons : NeuronGroup
            The instantiated neuron group with the parameters specified.

        Comments:
        --------
        add these neurons to some list of neuron groups attached to this type as well, and on init add neuron groups to some buffer somewhere? IDK how that would work... look into it same as network baso
        I guess on inport of module import that buffer so it hangs out in the background...
        maybe add some list which captures all neurongroups made according to this template, perhaps include it in some mapping feature as well - say if we could add all
        If the network has been explicitly created we can do this with target, otherwise it gets added to the global stuff - add functionality later
        """
        # If target is provided, handle functionality for adding neurons to a target later
        if target:
            return

        # Retrieve the appropriate equation model for the neuron type
        model = equations.neuron_equations[type]

        # Create the neuron group with a threshold and reset condition
        neurons = NeuronGroup(
            N=int(self.size**2),
            model=model,
            threshold="v > V_threshold",
            reset=self.parameters.v_reset,
            name=f"{self.neuron_type}_layer_{layer}",
        )

        # Add additional parameters to the neuron group
        self.add_variables(neurons)
        return neurons

    def add_variables(self, neurons):
        """
        Adds parameters as variables to the neuron group.

        Parameters:
        -----------
        neurons : NeuronGroup
            The neuron group to which variables will be added.
        """

        # Assign membrane, conductance, and voltage-related parameters
        neurons.Cm = self.parameters.cm
        neurons.g_leak = self.parameters.g_leak
        neurons.V_rest = self.parameters.v_rest
        neurons.V_reset = self.parameters.v_reset
        neurons.V_reversal_e = self.parameters.v_reversal_e
        neurons.V_reversal_i = self.parameters.v_reversal_i
        neurons.sigma = self.parameters.sigma
        neurons.tau_m = self.parameters.tau_m

        # Add synaptic time constants where applicable
        if self.parameters.tau_ee:
            neurons.tau_ee = self.parameters.tau_ee
        if self.parameters.tau_ei:
            neurons.tau_ei = self.parameters.tau_ei
        if self.parameters.tau_ie:
            neurons.tau_ie = self.parameters.tau_ie
        if self.parameters.tau_ii:
            neurons.tau_ii = self.parameters.tau_ii

        # Optionally assign x and y coordinates if spatial mapping is necessary
        self.add_x_and_y(neurons)

    def add_x_and_y(self, neurons):
        """
        Assigns x and y coordinates to neurons in the grid (if necessary).
        The actual implementation for spatial assignments is yet to be added.

        Parameters:
        -----------
        neurons : NeuronGroup
            The neuron group for which x, y coordinates are to be set.
        """
        pass


def create_synapses():
    """
    Creates synapses between neuron groups.
    Currently works for networks with excitatory -> inhibitory,
    inhibitory -> excitatory, and feedforward connections between excitatory neurons in different layers.

    To be implemented with further synaptic models.
    """
    pass


def create_neuron_groups(
    n_layers,
    input_neuron_specs: NeuronSpecs,
    exc_neuron_specs: NeuronSpecs,
    inh_neuron_specs: NeuronSpecs,
):
    """
    Creates neuron groups for each layer, including input, excitatory, and inhibitory neurons.

    Parameters:
    -----------
    n_layers : int
        Number of layers in the network.
    input_neuron_specs : NeuronSpecs
        Specifications for the input neuron group.
    exc_neuron_specs : NeuronSpecs
        Specifications for the excitatory neuron group.
    inh_neuron_specs : NeuronSpecs
        Specifications for the inhibitory neuron group.
    """
    # Iterate over each layer and create neurons based on their types
    for layer in n_layers:
        # create the input layer for layer 0
        if layer == 0:
            input_neuron_specs.create_neurons(layer)

        # Create excitatory and inhibitory neuron groups for each layer
        exc_neuron_specs.create_neurons(layer)
        inh_neuron_specs.create_neurons(layer)


# Create Network Function:
def create_network():
    """
    Creates a VisNet model by setting up multiple neuron layers and synapses.

    n_layers : int
        Number of layers to create in the network.
    exc_neuron_specs, inh_neuron_specs, input_neuron_specs : NeuronSpecs
        Specifications for excitatory, inhibitory, and input neurons.
    stdp_parameters : dict
        Synaptic parameters for Spike-Timing Dependent Plasticity (STDP).
    """

    n_layers = 3  # Number of layers to create

    # Define neuron specifications for excitatory neurons (needs initialization)
    exc_neuron_specs = NeuronSpecs()

    # Define neuron specifications for inhibitory neurons (needs initialization)
    inh_neuron_specs = NeuronSpecs()

    # Define neuron specifications for input neurons (needs initialization)
    input_neuron_specs = NeuronSpecs()

    # Define a dictionary to hold STDP parameters for synapses (needs further setup)
    stdp_parameters = {}

    # Call the function to create the neuron groups across layers

    create_neuron_groups(
        n_layers,
        exc_neuron_specs,
        inh_neuron_specs,
        input_neuron_specs,
        stdp_parameters,
    )

    # Placeholder for synapse creation in the network

    create_synapses()
