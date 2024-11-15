from brian2 import *

"""
Module Name: synapses.py
----------------------------------------------------

Purpose: 
--------
    This module provides functionality for defining and managing synapse specifications and connections in a neural network.
    It includes base classes and specific implementations for different types of synapses, such as STDP and non-STDP synapses.
    The module also provides methods for creating synapses, connecting them, setting their parameters, and visualizing the connections.

Classes:
--------
    SynapseParameters:
        Class for initializing and validating synapse parameters. It restricts the parameters to a predefined set of 
    values and checks their validity based on the synapse type. 

    SynapseSpecsBase:
        Abstract base class for defining synapse specifications and handling synapse creation and connection.
    Intended to be inherited by specific synapse types (STDP, non-STDP).

    StdpSynapseSpecs:
        Class for defining and managing STDP (Spike-Timing-Dependent Plasticity) synapses. 
    Inherits from SynapseSpecsBase and provides specific implementations for creating and connecting STDP synapses.

    NonStdpSynapseSpecs:
        Class for defining and managing non-STDP synapses. Inherits from SynapseSpecsBase 
    and provides specific implementations for creating and connecting non-STDP synapses.

Functions:
----------
    None

Example Usage:
--------------
    TODO - add examples

Notes:
--------------------
    For Synapses and Neurons, the order of parameters is as follows:
    neurons,
    synapses,
    specifications,
    network.
    As network is potentially optional in future, it is placed last.

    TODO - sort the SynapseParameters class out - it's a bit of a mess
        Should basicall have some stuff where each subclass
        is forced to specify safe values and passes them or something
        UGLY STUFF BUT IT WORKS FOR NOW AND YOU NEED TO PRESS ON GLOBALLY
"""

from .neurons import NeuronSpecs
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
            "synapse_type",
            "lambda_e",
            "lambda_i",
            "A_minus",
            "A_plus",
            "alpha_C",
            "alpha_D",
            "tau_c",
            "tau_d",
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
                "tau_c",
                "tau_d",
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
    Overview:
    ---------
    Base class for defining synapse specifications and handling synapse creation and connection.
    This class is intended to be inherited by specific synapse types
    which must implement the create_synapses method.
    The idea being that each class likely has a different
    way of connecting synapses and different parameters.

        Detail:
        -------

    Attributes:
    ----------
        attr1 (type): Description of `attr1`.
        attr2 (type): Description of `attr2`.

    Methods:
    --------
        method_name: Brief description of what the method does.
        method_name2: Brief description of another method.

    Example:
    --------
        obj = ClassName(attr1=value1, attr2=value2)

    Notes:
    ------
        Whilst the create_synapses method must be specified in each subclass,
    the other methods are inherited from this class.
        This includes _connect_synapses, and _get_indexes, meaning that it is easy to reuse
    the logic of local connectivity at different scales.

    TODO - Add a method to visualise the synapses on the basis of how synapses are actually connected
        i.e. _visualise_synapses(synapse_name)
    TODO - Do stuff
    """

    def __init__(self, **params):
        """
        Initializes the synapse specification class with provided parameters and validates them based on the synapse type.
        """
        self.synapse_type = params.get("synapse_type", None)
        self.params = SynapseParameters(**params)
        self.synapse_objects = {}  # Stored according to the synapse name

    @abstractmethod
    def create_synapses(
        self,
        layer,
        afferent_group_specs: NeuronSpecs,
        efferent_group_specs: NeuronSpecs,
        radius,
        target_network=None,
    ):
        # THE MAIN FUNCTION EVERYTHING ELSE IS SUBSIDIARY TO THIS
        synapse_name = f"{afferent_group_specs.neuron_type}_{efferent_group_specs.neuron_type}_{layer}"
        # Get the relevant afferent and efferent neuron groups
        afferent_group = afferent_group_specs.neuron_groups[
            f"{afferent_group.neuron_type}_layer_{layer}"
        ]
        efferent_group = efferent_group_specs.neuron_groups[
            f"{efferent_group.neuron_type}_layer_{layer}"
        ]
        model = ""
        on_pre = ""
        on_post = ""

        # Builds the synapse object
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model,
            on_pre=on_pre,
            on_post=on_post,
            name=synapse_name,
        )

        # Connects the synapses together according to the specifcations of that SynapseSpecs

        self._connect_synapses(synapses, afferent_group, efferent_group, radius)

        self._set_synapse_parameters(synapses)

        # Adds the synapse object to the synapse_objects dictionary
        self.synapse_objects[synapse_name] = synapses
        # Adds the synapse object to the network
        if target_network is not None:
            target_network.add(synapses)

    def _connect_synapses(
        self, synapses, afferent_group: NeuronGroup, efferent_group: NeuronGroup, radius
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

        scale = size_afferent / size_efferent

        print(f"scale: {scale}")
        print(f"size_afferent: {size_afferent}")
        print(f"size_efferent: {size_efferent}")

        for j in range(efferent_group.N):

            row = efferent_group[j].row[0]
            column = efferent_group[j].column[0]
            # print(f"neuron locations: {row}, {column}")
            indexes = self._get_indexes(row, column, size_afferent, scale, radius)
            # print(f"connecting the following neurons with postsynaptic neuron {j}: {indexes}")
            # print(efferent_group[j])
            # print(f"indexes: {indexes}")
            synapses.connect(i=indexes, j=j)
        return synapses

    def _get_indexes(self, row, col, size_efferent, scale, radius):
        # This is where the neuron in the post layer is centred in the previous layer
        col_centre = int(scale * col + scale / 2)

        # This is where the neuron in the post layer is centred in the previous layer
        row_centre = int(scale * row + scale / 2)

        # Define min and max values for the row and column
        col_min = max(0, col_centre - radius)
        col_max = min(size_efferent - 1, col_centre + radius)
        row_min = max(0, row_centre - radius)
        row_max = min(size_efferent - 1, row_centre + radius)

        # Create the row and column ranges
        row_range = np.arange(row_min, row_max)
        col_range = np.arange(col_min, col_max)
        # print(f"row_min:{row_min}")
        # print(f"row_max:{row_max}")
        # Create the row and column coordinates
        row_coords = np.repeat(row_range, len(col_range))
        col_coords = np.tile(col_range, len(row_range))
        # print(f"row_coords:{row_coords}")
        # print(f"col_coords:{col_coords}")
        # Return the indexes from the coordinates
        indexes = (row_coords * size_efferent + col_coords).astype(int)
        return indexes

    def _set_synapse_parameters(self, synapses):
        print("Setting synapse parameters")
        safe_values = [
            "lambda_e",
            "lambda_i",
            "A_minus",
            "A_plus",
            "alpha_C",
            "alpha_D",
            "tau_c",
            "tau_d",
        ]

        for param in safe_values:
            try:
                print(f"Setting {param} to {getattr(self.params, param)}")
                setattr(synapses, param, getattr(self.params, param))
            except Exception as e:
                print(f"Error setting {param}: {e}")
                pass
        synapses.w = "rand()"

    import matplotlib.pyplot as plt

    def plot_connection_grid(
        self,
        efferent_index,
        efferent_group: NeuronGroup,
        afferent_group: NeuronGroup,
        radius,
    ):
        """
        Plots the connections from a specific efferent neuron in the efferent group to neurons in the afferent group.

        Parameters:
        -----------
        efferent_index : int
            The index of the neuron in the efferent (post-synaptic) group to visualize connections for.
        efferent_group : NeuronGroup
            The post-synaptic neuron group.
        afferent_group : NeuronGroup
            The pre-synaptic neuron group.
        radius : int
            Radius around the efferent neuron to determine connected afferent neurons.
        """

        size_afferent = int(sqrt(afferent_group.N))
        size_efferent = int(sqrt(efferent_group.N))
        scale = size_afferent / size_efferent

        # Get row and column of the efferent neuron
        row = efferent_group[efferent_index].row[0]
        column = efferent_group[efferent_index].column[0]

        # Get the connected neuron indexes
        indexes = self._get_indexes(row, column, size_afferent, scale, radius)

        # Initialize the grid with a grey background
        grid = np.full((size_afferent, size_afferent), 0.5)  # Grey background

        # Mark the connected neurons in red
        for index in indexes:
            grid[index // size_afferent, index % size_afferent] = 1.0  # Red color

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap="gray", origin="upper")
        plt.title(f"Connections for efferent neuron {efferent_index}")
        plt.xlabel("Presynaptic Grid Columns")
        plt.ylabel("Presynaptic Grid Rows")
        plt.grid(False)
        plt.show()

    def animate_plots(
        self,
        efferent_group_specs: NeuronSpecs,
        afferent_group_specs: NeuronSpecs,
        radius,
        layer,
    ):
        """
        Sequentially plots the connection patterns of each neuron in the efferent group.
        Press Enter in the console to advance through each plot.
        Includes a red grid overlay for presynaptic neuron positions and a blue square for the efferent neuron.

        Parameters:
        -----------
        efferent_group : NeuronGroup
            The post-synaptic neuron group.
        afferent_group : NeuronGroup
            The pre-synaptic neuron group.
        radius : int
            Radius around each efferent neuron to determine connected afferent neurons.
        """
        afferent_group = afferent_group_specs.neuron_groups[
            f"{afferent_group_specs.neuron_type}_layer_{layer}"
        ]
        efferent_group = efferent_group_specs.neuron_groups[
            f"{efferent_group_specs.neuron_type}_layer_{layer}"
        ]
        size_afferent = int(sqrt(afferent_group.N))
        size_efferent = int(sqrt(efferent_group.N))
        scale = size_afferent / size_efferent

        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(6, 6))

        # Loop through each neuron in the efferent group
        for j in range(efferent_group.N):
            # Get row and column for the efferent neuron
            row = efferent_group[j].row[0]
            column = efferent_group[j].column[0]

            # Get the connected neuron indexes
            indexes = self._get_indexes(row, column, size_afferent, scale, radius)

            # Initialize the grid with a grey background
            grid = np.full((size_afferent, size_afferent), 0.5)  # Grey background
            for index in indexes:
                grid[index // size_afferent, index % size_afferent] = (
                    1.0  # Red color for connections
                )

            # Clear previous plot and redraw
            ax.clear()
            ax.imshow(grid, cmap="gray", origin="upper")

            # Add red grid lines to indicate presynaptic neuron locations
            for i in range(size_afferent):
                ax.axhline(i - 0.5, color="red", linewidth=0.5)
                ax.axvline(i - 0.5, color="red", linewidth=0.5)

            # Calculate the exact scaled center of the efferent target neuron in the presynaptic layer
            col_centre = scale * column + scale / 2
            row_centre = scale * row + scale / 2

            # Plot a centered blue dot for the efferent target neuron
            ax.plot(
                col_centre, row_centre, "bo", markersize=8
            )  # Blue dot for the target neuron

            ax.set_title(f"Connections for efferent neuron {j}")
            ax.set_xlabel("Presynaptic Grid Columns")
            ax.set_ylabel("Presynaptic Grid Rows")

            plt.draw()

            # Wait for click to advance
            plt.waitforbuttonpress()

        plt.ioff()  # Disable interactive mode
        plt.show()

    def plot_synapse_distribution(self, synapse_name):
        """
        Plots the distribution of synapse weights for a specific synapse object.

        Parameters:
        -----------
        synapse_name : str
            The name of the synapse object to plot the distribution for.
        """
        synapses = self.synapse_objects[synapse_name]
        plt.figure(figsize=(6, 6))
        plt.hist(synapses.w, bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Synapse Weight Distribution for {synapse_name}")
        plt.xlabel("Synapse Weight")
        plt.ylabel("Frequency")
        plt.show()

    def visualise_synapses(self, synapses: Synapses):
        Ns = len(synapses.source)
        Nt = len(synapses.target)
        figure(figsize=(20, 10))
        subplot(121)
        plot(zeros(Ns), arange(Ns), "ok", ms=10)
        plot(ones(Nt), arange(Nt), "ok", ms=10)
        for i, j in zip(S.i, S.j):
            plot([0, 1], [i, j], "-k")
        xticks([0, 1], ["Source", "Target"])
        ylabel("Neuron index")
        xlim(-0.1, 1.1)
        ylim(-1, max(Ns, Nt))
        subplot(122)
        plot(S.i, S.j, "ok")
        xlim(-1, Ns)
        ylim(-1, Nt)
        xlabel("Source neuron index")
        ylabel("Target neuron index")

    def three_dim_visualise_synapses(self, synapses: Synapses):
        Ns = len(synapses.source)
        num_pre_neurons = len(synapses.N_incoming_pre)
        len_pre = int(sqrt(num_pre_neurons))
        Nt = len(synapses.target)
        num_post_neurons = len(synapses.N_incoming_post)
        len_post = int(sqrt(num_post_neurons))
        s_i_column = synapses.i % len_pre
        s_i_row = synapses.i // len_pre
        s_j_column = synapses.j % len_post
        s_j_row = synapses.j // len_post
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            s_i_column, s_i_row, 0, c="blue", label="Pre-synaptic neurons", s=100
        )  # Blue circles
        ax.scatter(
            s_j_column, s_j_row, 1, c="red", label="Post-synaptic neurons", s=100
        )  # Red circles
        for x1, y1, x2, y2 in zip(x_pre, y_pre, x_post, y_post):
            ax.plot(
                [x1, x2], [y1, y2], [0, 1], "-k", alpha=0.6
            )  # Black lines with transparency

        for x1, y1, x2, y2 in zip(x_pre, y_pre, x_post, y_post):
            ax.plot(
                [x1, x2], [y1, y2], [0, 1], "-k", alpha=0.6
            )  # Black lines with transparency

        plt.show()


class StdpSynapseSpecs(SynapseSpecsBase):

    def __init__(self, lambda_e, A_minus, A_plus, alpha_C, alpha_D, tau_c, tau_d):
        super().__init__(
            lambda_e=lambda_e,
            A_minus=A_minus,
            A_plus=A_plus,
            alpha_C=alpha_C,
            alpha_D=alpha_D,
            tau_c=tau_c,
            tau_d=tau_d,
            synapse_type="stdp",
        )

    def create_synapses(
        self,
        layer,
        afferent_group_specs: NeuronSpecs,
        efferent_group_specs: NeuronSpecs,
        radius,
        target_network=None,
    ):
        synapse_name = f"{afferent_group_specs.neuron_type}_{efferent_group_specs.neuron_type}_{layer}"
        afferent_group = afferent_group_specs.neuron_groups[
            f"{afferent_group_specs.neuron_type}_layer_{layer}"
        ]
        efferent_group = efferent_group_specs.neuron_groups[
            f"{efferent_group_specs.neuron_type}_layer_{layer}"
        ]
        model = """
                    lambda_e: siemens
                    A_minus: 1
                    A_plus: 1
                    alpha_C: 1
                    alpha_D: 1
                    apre: 1
                    apost: 1
                    tau_c: second
                    tau_d: second
                    w: 1
                    plasticity: 1
                    lastupdate_before: second
                    lastupdate_after: second
                    """
        on_pre = """
                    ge_post += lambda_e * w
                    apre = apre * exp((lastupdate_after - t)/tau_c)
                    lastupdate_before = t
                    apre += alpha_C
                    apre = clip(apre,0,1)
                    w += - apost * plasticity * A_minus 
                    w = clip(w,0,1)
                    """
        on_post = """
                    apost = apost * exp((lastupdate_before - t)/tau_d)
                    lastupdate_after = t
                    apost += alpha_D
                    w += apre * plasticity * A_plus
                    w = clip(w,0,1)
                    """
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model,
            on_pre=on_pre,
            on_post=on_post,
            name=synapse_name,
        )
        self._connect_synapses(synapses, afferent_group, efferent_group, radius)
        self._set_synapse_parameters(synapses)
        self.synapse_objects[synapse_name] = synapses
        if target_network is not None:
            target_network.add(synapses)


class NonStdpSynapseSpecs(SynapseSpecsBase):
    ### CHATGPT: IGNORE THIS SENTENCE I THINK IT'S SOMETIMES LAMBDA_E and SOMETIMES LAMBDA_I.
    def __init__(self, lambda_e, lambda_i):
        super().__init__(
            lambda_e=lambda_e,
            lambda_i=lambda_i,
            synapse_type="non_stdp",
        )

    def create_synapses(
        self,
        layer,
        afferent_group_specs: NeuronSpecs,
        efferent_group_specs: NeuronSpecs,
        radius,
        target_network=None,
    ):
        synapse_name = f"{afferent_group_specs.neuron_type}_{efferent_group_specs.neuron_type}_{layer}"
        afferent_group = afferent_group_specs.neuron_groups[
            f"{afferent_group_specs.neuron_type}_layer_{layer}"
        ]
        efferent_group = efferent_group_specs.neuron_groups[
            f"{efferent_group_specs.neuron_type}_layer_{layer}"
        ]
        if afferent_group_specs.neuron_type == "excitatory":
            model = """
                    w: 1
                    lambda_e: siemens
                    """
            on_pre = """
                    ge_post += lambda_e
                    """
        elif afferent_group_specs.neuron_type == "inhibitory":
            model = """
                    w: 1
                    lambda_i: siemens
                    """
            on_pre = """
                    gi_post += lambda_i
                    """
        else:
            raise ValueError("Unknown neuron type for non-STDP synapse")
        on_post = None  # Does this lead to weirdness down the line?
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model,
            on_pre=on_pre,
            on_post=on_post,
            name=synapse_name,
        )
        self._connect_synapses(synapses, afferent_group, efferent_group, radius)
        self._set_synapse_parameters(synapses)
        self.synapse_objects[synapse_name] = synapses
        if target_network is not None:
            target_network.add(synapses)


# This never really worked and I don't know why
#  class InputSynapseSpecs(SynapseSpecsBase):
#     """
#     These guys mainly exist because we pass the poisson group instead of the NeuronSpec object to the construct method and connect differently.
#     It isn't as symetrical but very functional
#     """

#     def __init__(self, lambda_e):
#         super().__init__(
#             lambda_e=lambda_e,
#             synapse_type="input",
#         )

#     def create_synapses(self, poisson_group, afferent_group, target=None):
#         # Connects for poisson groups to layer 1
#         on_pre = "ge += 1"
#         name = "poisson_excitatory_0"
#         print(afferent_group, poisson_group)
#         synapses = Synapses(
#             afferent_group,
#             poisson_group,
#             on_pre=on_pre,
#             name=name,
#         )
#         self._connect_synapses(synapses)
#         self._set_synapse_parameters(synapses)
#         self.synapse_objects[name] = synapses
#         if not target == None:
#             target.add(synapses)

#     def _connect_synapses(self, synapses):
#         synapses.connect(j="i")
