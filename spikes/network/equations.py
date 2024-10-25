from brian2 import *


# OKAY THIS WHOLE THING ISN"T RIGHT - SINGLE EQUATION vs EQUATIONS?
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
        """
        Initialize empty dictionaries for neuron, synaptic, and other equations.
        """
        # Initialise the dictionaries for different equations
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
            column : integer (constant)
            row : integer (constant)
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
            column: integer (constant)
            row: integer (constant)
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
            column : integer (constant)
            row : integer (constant)
            """
        )
        stdp_model = Equations(
            """
                    lambda_e: 1
                    A_minus: 1
                    A_plus: 1
                    alpha_C: 1
                    alpha_D: 1
                    apre: 1
                    apost: 1
                    w: 1
                    lastupdate_pre: second
                    lastupdate_post: second
                    """
        )
        stdp_pre = Equations(
            """
                    ge_post += lambda_e * w
                    apre = apre * exp((lastupdate_post - t)/tau_pre)
                    lastupdate_pre = t
                    apre += alpha_C
                    apre = clip(apre,0,1)
                    w += - apost * A_minus 
                    w = clip(w,0,1)
                    """
        )
        stdp_post = Equations(
            """
                    apost = apost * exp((lastupdate_pre - t)/tau_post)
                    lastupdate_post = t
                    apost += alpha_D
                    w += apre * A_plus
                    w = clip(w,0,1)
                    """
        )
        non_stdp_model = Equations(" ")

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
