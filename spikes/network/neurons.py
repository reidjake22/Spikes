from brian2 import *

# neurons.py
"""
Module Name: neurons.py
----------------------------------------------------

Purpose: 
--------
    This module provides classes and functions for defining and managing neuron parameters and specifications.
    It includes functionality for validating neuron parameters, creating neuron groups, and adding variables to neuron groups.

Classes:
--------
    NeuronParameters:
        Class to hold and validate neuron parameters such as membrane capacitance, leak conductance, firing threshold voltage, etc.

    NeuronSpecs:
        Class to hold neuron specifications such as type, length, and corresponding parameters. It includes methods for creating neurons and adding variables to neuron groups.

Variables:
----------
    equations (EquationsContainer):
        An instance of EquationsContainer used to store global equations for neurons.

Example Usage:
--------------
    TODO - add examples

Notes:
--------------------
    TODO - add notes
"""

from .equations import EquationsContainer
import warnings
import seaborn as sns

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
        v_threshold=None,
        v_reset=None,
        v_rest=None,
        v_reversal_e=None,
        v_reversal_i=None,
        v_reversal_a=None,
        t_refract=None,
        sigma=None,
        tau_m=None,
        tau_ee=None,
        tau_ei=None,
        tau_ie=None,
        tau_ii=None,
        tau_a=None,
        neuron_type=None,
    ):
        # Assign validated parameters to the class attributes
        self.cm = cm
        self.g_leak = g_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.v_reversal_e = v_reversal_e
        self.v_reversal_i = v_reversal_i
        self.v_reversal_a = v_reversal_a
        self.t_refract = t_refract
        self.sigma = sigma
        self.tau_m = tau_m
        self.tau_a = tau_a
        for tau_name, tau_value in [
            ("tau_ee", tau_ee),
            ("tau_ei", tau_ei),
            ("tau_ie", tau_ie),
            ("tau_ii", tau_ii),
            ("tau_a", tau_a),
        ]:
            setattr(self, tau_name, tau_value)  # Assign the attribute
            # print(f"Assigned {tau_name}={tau_value}")  # Debugging output
        # Assign a list to keep track of all groups made with these specs
        self.check_valid_parameters(neuron_type)

    def check_valid_parameters(self, neuron_type):
        required_params = [
            "cm",
            "g_leak",
            "v_threshold",
            "v_reset",
            "v_rest",
            "v_reversal_e",
            "v_reversal_i",
            "v_reversal_a",
            "t_refract",
            "sigma",
            "tau_m",
        ]
        params_by_type = {
            "e": {
                "required": ["tau_ee", "tau_ie", "tau_a"],
                "forbidden": ["tau_ei", "tau_ii"],
            },
            "i": {
                "required": ["tau_ei", "tau_ii"],
                "forbidden": ["tau_ee", "tau_ie", "tau_a"],
            },
        }

        # Ensure all required parameters are provided (i.e., not None)
        missing_params = [p for p in required_params if getattr(self, p) is None]
        if missing_params:
            warnings.warn(f"Missing required parameters: {', '.join(missing_params)}")

        if neuron_type is None:
            raise ValueError("Neuron type must be specified.")
        # Explicitly check if excitatory parameters are consistent
        if neuron_type in params_by_type:
            required = params_by_type[neuron_type]["required"]
            forbidden = params_by_type[neuron_type]["forbidden"]
            inconsistent = any(getattr(self, p) is None for p in required) or any(
                getattr(self, p) for p in forbidden
            )
            if inconsistent:
                warnings.warn(
                    f"Mismatch in conductance parameters for type '{neuron_type}'. "
                    f"Required: {', '.join(required)}; Forbidden: {', '.join(forbidden)}. "
                    f"Check values: {', '.join(f'{p}={getattr(self, p)}' for p in required + forbidden)}"
                )
        else:
            raise ValueError(f"Invalid neuron_type: {neuron_type}. Must be 'e' or 'i'.")


class NeuronSpecs:
    """
    Class to hold neuron specifications such as type, length, and corresponding parameters.

    Parameters:
    -----------
    neuron_type : str
        Type of the neuron ('excitatory', 'inhibitory', etc.).
    length : int
        length of the neuron group (e.g., grid dimensions for spatially organized groups).
    cm, g_leak, v_threshold, etc. : various types
        Neuron parameters required to initialize the group.
    """

    def __init__(
        self,
        neuron_type,
        length,
        cm=None,
        g_leak=None,
        v_threshold=None,
        v_reset=None,
        v_rest=None,
        v_reversal_e=None,
        v_reversal_i=None,
        v_reversal_a=None,
        t_refract=None,
        sigma=None,
        tau_m=None,
        tau_ee=None,
        tau_ei=None,
        tau_ie=None,
        tau_ii=None,
        tau_a=None,
    ):
        # Store neuron type and length, and validate neuron parameters
        self.neuron_type = neuron_type
        self.length = length
        self.parameters = NeuronParameters(
            cm,
            g_leak,
            v_threshold,
            v_reset,
            v_rest,
            v_reversal_e,
            v_reversal_i,
            v_reversal_a,
            t_refract,
            sigma,
            tau_m,
            tau_ee,
            tau_ei,
            tau_ie,
            tau_ii,
            tau_a,
            neuron_type,
        )
        self.neuron_groups = (
            {}
        )  # Store neuron groups created with these specs accessible by layer number

    def add_neurons(self, layer, neuron_group):
        """
        Adds a neuron group to the list of neuron groups created with these specs.

        Parameters:
        -----------
        layer : int
            The layer number to which the neuron group belongs.
        neuron_group : NeuronGroup
            The neuron group to be added.
        """
        if self.neuron_groups.get(layer):
            warnings.warn(f"Neuron group for layer {layer} already exists.")
        else:
            self.neuron_groups[layer] = neuron_group

    def create_neurons(self, layer, target_network=None):
        """
        Creates neurons in the given layer.

        Parameters:
        -----------
        layer : int
            The current layer number where neurons should be created.
        target_network : optional
            The target_network network to which the neurons should be added (optional).

        Returns:
        --------
        neurons : NeuronGroup
            The instantiated neuron group with the parameters specified.

        Comments:
        --------
        add these neurons to some list of neuron groups attached to this type as well,
        and on init add neuron groups to some buffer somewhere? IDK how that would work... look into it same as network baso
        I guess on inport of module import that buffer so it hangs out in the background...
        maybe add some list which captures all neurongroups made according to this template,
          perhaps include it in some mapping feature as well - say if we could add all
        If the network has been explicitly created we can do this with target_network,
        otherwise it gets added to the global stuff - add functionality later
        """

        # Retrieve the appropriate equation model for the neuron type
        model = equations.neuron_equations[self.neuron_type]
        if self.neuron_type == "e":
            reset = """
            v = V_reset
            ga += 6 * nsiemens """
        else:
            reset = """
            v = V_reset
            """
        # print(f"for neuron type {self.neuron_type} using model {model}")
        neuron_group_name = f"{self.neuron_type}_{layer}"
        t_refract = self.parameters.t_refract
        # Create the neuron group with a threshold and reset condition
        neurons = NeuronGroup(
            N=int(self.length**2),
            model=model,
            method="rk4",
            threshold="v > V_threshold",
            reset=reset,
            refractory=t_refract,
            name=neuron_group_name,
        )
        # Add additional parameters to the neuron group
        self.add_variables(neurons)
        self.neuron_groups[layer] = neurons
        if not target_network == None:
            print(f"adding layer{layer} neurons to network")
            target_network.add(neurons)
        return neurons

    def add_variables(self, neurons):
        """
        Adds parameters as variables to the neuron group.

        Parameters:
        -----------
        neurons : NeuronGroup
            The neuron group to which variables will be added.
        """
        neurons.v = self.parameters.v_rest

        # Assign membrane, conductance, and voltage-related parameters
        neurons.Cm = self.parameters.cm
        neurons.g_leak = self.parameters.g_leak
        neurons.V_rest = self.parameters.v_rest
        neurons.V_reset = self.parameters.v_reset
        neurons.V_reversal_e = self.parameters.v_reversal_e
        neurons.V_reversal_i = self.parameters.v_reversal_i
        neurons.V_threshold = self.parameters.v_threshold
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
        if self.parameters.tau_a:
            neurons.tau_a = self.parameters.tau_a
        if self.parameters.v_reversal_a:
            neurons.V_reversal_a = self.parameters.v_reversal_a

        # Optionally assign x and y coordinates if spatial mapping is necessary
        self.add_rows_and_columns(neurons, self.length)

    def add_rows_and_columns(self, neurons, length):
        """
        Assigns rows and columns to neurons in the grid (if necessary).
        The actual implementation for spatial assignments is yet to be added.

        Parameters:
        -----------
        neurons : NeuronGroup
            The neuron group for which indexes are to be set.

        row, column coordinates look like this:
        0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16
        0,0 0,1 0,2 0,3 0,4 1,2 2,2 3,2 4,2 1,3 2,3 3,3 4,3 1,4 2,4 3,4 4,4

        so arranged in an cartesian fashion it would look as follows:

        13  14  15  16
        9   10  11  12
        5   6   7   8
        1   2   3   4

        However what we're doing is working with it in a ML context so it really looks like this:
                Column
                1     2       3       4
            1   1     2       3       4

        Row 2   5     6       7       8

            3   9     10      11      12

            4   13    14      15      16
        """
        columns = np.tile(np.arange(length), length)
        rows = np.repeat(np.arange(length), length)
        neurons.column = columns
        neurons.row = rows
