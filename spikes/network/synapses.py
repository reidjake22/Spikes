from brian2 import *
from neurons import NeuronSpecs
from abc import ABC, abstractmethod


class SynapseParameters:
    def __init__(self, **params):
        """
        Initialize the synapse parameters with the provided values and check if they are valid.
        Restricts the parameters to a predefined set of values:
        - lambda_e
        - A_minus
        - A_plus
        - alpha_C
        - alpha_D
        - tau_pre
        - tau_post
        """
        safe_values = [
            "lambda_e",
            "lambda_i",
            "A_minus",
            "A_plus",
            "alpha_C",
            "alpha_D",
            "tau_pre",
            "tau_post",
        ]
        for key, value in params.items():
            if key not in safe_values:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        self.check_valid_parameters()

    def check_valid_parameters(self):
        """
        ChatGPT ignore this line: Add to as new subclasses of SynapseSpec are created - this feels buggy but idk what to do otherwise - maybe add into the logic of each subclass or add it as a sort of custom class to override each time if needed and then run in the base?
        """
        if self.synapse_type == "stdp":
            for key in [
                "lambda_e",
                "A_minus",
                "A_plus",
                "alpha_C",
                "alpha_D",
                "tau_pre",
                "tau_post",
            ]:
                if not hasattr(self, key) or getattr(self, key) is None:
                    raise ValueError(f"Parameter {key} is not provided")
        elif self.synapse_type == "non_stdp":
            if not hasattr(self, "lambda_e") or self.lambda_e is None:
                raise ValueError(
                    "lambda_e is not provided, this is needed for non_stdp synapses"
                )
            if not hasattr(self, "lambda_i") or self.lambda_i is None:
                raise ValueError(
                    "lambda_i is not provided, this is needed for non_stdp synapses"
                )


class SynapseSpecsBase(ABC):
    """
    Base class for defining synapse specifications and handling synapse creation and connection.
    This class is intended to be inherited by specific synapse types (STDP, non-STDP)
    attributes:
        params (SynapseParameters): An object which stores and verifies synaptic parameters
    """

    def __init__(self, **params):
        """
        Initializes the synapse specification class with provided parameters and validates them based on the synapse type.
        """
        self.synapse_type = params.get("synapse_type", None)
        self.params = SynapseParameters(**params)
        self.synapse_objects = []

    @abstractmethod
    def construct(
        self,
        layer,
        afferent_group: NeuronSpecs | PoissonGroup,
        efferent_group: NeuronSpecs,
    ):
        """
        Abstract method to construct synapses between afferent and efferent neuron groups.
        Will handle the main logic of creating synapses and connecting them.
        the following is a general outline of what the method should do:
        """
        model = ""
        on_pre = ""
        on_post = ""
        synapses = self.create_synapses(
            layer, afferent_group, efferent_group, model, on_pre, on_post
        )
        self.synapse_objects.append(synapses)
        self.connect_synapses(synapses, efferent_group, afferent_group)

    def set_synapse_parameters(self, synapses):
        safe_values = [
            "lambda_e",
            "A_minus",
            "A_plus",
            "alpha_C",
            "alpha_D",
            "tau_pre",
            "tau_post",
        ]
        for param in safe_values:
            setattr(synapses, param, getattr(self.params, param))

    def create_synapses(
        self,
        layer,
        afferent_group: NeuronSpecs,
        efferent_group: NeuronSpecs,
        model,
        on_pre,
        on_post=None,
    ):
        """
        Creates the synapses between the afferent (pre-synaptic) and efferent (post-synaptic) neuron groups.

        Args:
            afferent_group (NeuronGroup): The group of pre-synaptic neurons.
            efferent_group (NeuronGroup): The group of post-synaptic neurons.
            model (str): The model definition string for the synapse (includes synapse variables and their equations).
            on_pre (str): The event handling string for the pre-synaptic neuron firing.
            on_post (str, optional): The event handling string for the post-synaptic neuron firing (only for STDP).

        Returns:
            Synapses: The created synapse object with the specified model and event handling.
        """
        afferent_type = afferent_group.neuron_type
        efferent_type = efferent_group.neuron_type
        name = f"{afferent_type}_{efferent_type}_{layer}"
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model,
            on_pre=on_pre,
            on_post=on_post,
            name=name,
        )
        self.set_synapse_parameters(synapses)
        return synapses

    def connect_synapses(
        self, synapses, efferent_group: NeuronSpecs, afferent_group: NeuronSpecs, radius
    ):
        """
        Connects the synapses between the afferent and efferent neuron groups.

        Parameters:
        -----------
        synapses : Synapses
            The synapse object to connect.
        efferent_group : NeuronGroup
            The post-synaptic neuron group.
        afferent_group : NeuronGroup
            The pre-synaptic neuron group.
        """
        size_afferent = sqrt(afferent_group.N)
        size_efferent = sqrt(efferent_group.N)
        scale = size_afferent // size_efferent
        for j in range(efferent_group.N):
            row = efferent_group[j].row
            column = efferent_group[j].column
            indexes = self.get_indexes(self, row, column, size_afferent, scale, radius)
            synapses.connect(i=indexes, j=j)
        return synapses

    def get_indexes(self, row_coords, col_coords, size_afferent, scale, radius):
        self.get_sources(row_coords, col_coords, size_afferent, scale, radius)
        return int(row_coords * size_afferent + col_coords)

    def get_sources(self, row, column, size_afferent, scale, radius):
        col_centre = int(
            scale * column + scale / 2
        )  # This is where the neuron in the post layer is centred in the previous layer
        row_centre = int(
            scale * row + scale / 2
        )  # This is where the neuron in the post layer is centred in the previous layer
        col_min = max(0, col_centre - radius)
        col_max = min(size_afferent - 1, col_centre + radius)
        row_min = max(0, row_centre - radius)
        row_max = min(size_afferent - 1, row_centre + radius)
        row_range = np.arange(row_min, row_max)
        col_range = np.arange(col_min, col_max)
        row_coords = np.repeat(row_range, len(col_range))
        col_coords = np.tile(col_range, len(row_range))
        return row_coords, col_coords


class STDPSynapseSpecs(SynapseSpecsBase):

    def __init__(self, lambda_e, A_minus, A_plus, alpha_C, alpha_D, tau_pre, tau_post):
        super().__init__(
            lambda_e=lambda_e,
            A_minus=A_minus,
            A_plus=A_plus,
            alpha_C=alpha_C,
            alpha_D=alpha_D,
            tau_pre=tau_pre,
            tau_post=tau_post,
            synapse_type="stdp",
        )

    def construct(
        self,
        layer,
        afferent_group: NeuronSpecs,
        efferent_group: NeuronSpecs,
        target=None,
    ):
        """
        Constructs the STDP synapses between the afferent and efferent neuron groups."""
        model = """
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
        on_pre = """
                    ge_post += lambda_e * w
                    apre = apre * exp((lastupdate_post - t)/tau_pre)
                    lastupdate_pre = t
                    apre += alpha_C
                    apre = clip(apre,0,1)
                    w += - apost * A_minus 
                    w = clip(w,0,1)
                    """
        on_post = """
                    apost = apost * exp((lastupdate_pre - t)/tau_post)
                    lastupdate_post = t
                    apost += alpha_D
                    w += apre * A_plus
                    w = clip(w,0,1)
                    """
        afferent_type = afferent_group.neuron_type
        efferent_type = efferent_group.neuron_type
        name = f"{afferent_type}_{efferent_type}_{layer+1}"
        synapses = self.create_synapses(
            afferent_group, efferent_group, model, on_pre, on_post, name=name
        )
        self.synapse_objects.append(synapses)
        if target:
            target.add(synapses)
        self.connect_synapses(synapses, efferent_group, afferent_group)


class NonSTDPSynapseSpecs(SynapseSpecsBase):
    ### CHATGPT: IGNORE THIS SENTENCE I THINK IT'S SOMETIMES LAMBDA_E and SOMETIMES LAMBDA_I.
    def __init__(self, lambda_e, lambda_i):
        super().__init__(
            lambda_e=lambda_e,
            lambda_i=lambda_i,
            synapse_type="non_stdp",
        )

    def construct(
        self,
        layer,
        afferent_group: NeuronSpecs | PoissonGroup,
        efferent_group: NeuronSpecs,
        target=None,
    ):

        if afferent_group.neuron_type == "excitatory":
            model = """
                    w: 1
                    lambda_e: 1
                    """
            on_pre = """
                    ge_post += lambda_e * w
                    """
        elif afferent_group.neuron_type == "inhibitory":
            model = """
                    w: 1
                    lambda_i: 1
                    """
            on_pre = """
                    gi_post += lambda_i * w
                    """
        else:
            raise ValueError("Unknown neuron type for non-STDP synapse")

        on_post = None  # Does this lead to weirdness down the line?
        afferent_type = afferent_group.neuron_type
        efferent_type = efferent_group.neuron_type
        name = f"{afferent_type}_{efferent_type}_{layer+1}"

        synapses = self.create_synapses(
            afferent_group, efferent_group, model, on_pre, on_post, name=name
        )
        self.synapse_objects.append(synapses)
        if target:
            target.add(synapses)
        self.connect_synapses(synapses, efferent_group, afferent_group)


class InputSynapseSpecs(SynapseSpecsBase):
    """
    These guys mainly exist because we pass the poisson group instead of the NeuronSpec object to the construct method and connect differently.
    It isn't as symetrical but very functional
    """

    def __init__(self, lambda_e):
        super().__init__(
            lambda_e=lambda_e,
            synapse_type="input",
        )

    def construct(self, poisson_group, afferent_group, target=None):
        # Connects for poisson groups to layer 1
        model = """
                    w: 1
                    lambda_e: 1
                    """
        on_pre = """
                    ge_post += lambda_e * w
                    """
        on_post = None  # Does this lead to weirdness down the line?
        name = "poisson_excitatory_0"

        synapses = self.create_synapses(
            afferent_group, poisson_group, model, on_pre, on_post, name=name
        )
        self.synapse_objects.append(synapses)
        if target:
            target.add(synapses)
        self.connect_synapses(synapses, poisson_group, afferent_group)

    def connect_synapses(self, synapses, poisson_group, afferent_group):
        synapses.connect(j="i")
