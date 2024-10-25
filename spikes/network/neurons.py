from brian2 import *
from equations import EquationsContainer

# Initialize the global equations container | look into this
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
        neuron_type=None,
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

        # Assign a list to keep track of all groups made with these specs
        self.neuron_groups = []

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
    Class to hold neuron specifications such as type, length, and corresponding parameters.

    Parameters:
    -----------
    neuron_type : str
        Type of the neuron ('excitatory', 'inhibitory', etc.).
    length : int
        length of the neuron group (e.g., grid dimensions for spatially organized groups).
    cm, g_leak, v_leak, v_threshold, etc. : various types
        Neuron parameters required to initialize the group.
    """

    def __init__(
        self,
        neuron_type,
        length,
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
        # Store neuron type and length, and validate neuron parameters
        self.neuron_type = neuron_type
        self.length = length
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

        # Retrieve the appropriate equation model for the neuron type
        model = equations.neuron_equations[type]

        # Create the neuron group with a threshold and reset condition
        neurons = NeuronGroup(
            N=int(self.length**2),
            model=model,
            threshold="v > V_threshold",
            reset=self.parameters.v_reset,
            name=f"{self.neuron_type}_layer_{layer+1}",
        )
        # Add additional parameters to the neuron group
        self.add_variables(neurons)
        if target:
            target.add(neurons)
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
        self.neuron_groups.append(neurons)
        self.add_rows_and_columns(neurons)

    def add_rows_and_columns(self, neurons, length):
        """
        Assigns rows and columns to neurons in the grid (if necessary).
        The actual implementation for spatial assignments is yet to be added.

        Parameters:
        -----------
        neurons : NeuronGroup
            The neuron group for which indexes are to be set.
        """
        columns = np.tile(np.arange(length), length)
        rows = np.repeat(np.arange(length), length)
        neurons.column = columns
        neurons.row = rows
