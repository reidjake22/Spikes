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


class SynapseParameters:
    def __init__(self, **params):
        """
        Initialize the synapse parameters with the provided values and check if they are valid.
        Restricts the parameters to a predefined set of values:
        - type
        - lambda_e
        = lambda_i
        - A_plus
        - alpha_C
        - alpha_D
        - tau_pre
        - tau_post
        """
        safe_values = [
            "type",
            "lambda_e",
            "lambda_i",
            "A_plus",
            "alpha_C",
            "alpha_D",
            "tau_c",
            "tau_d",
            "learning_rate",
        ]
        for key, value in params.items():
            if key not in safe_values:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        self.check_valid_parameters()

    def check_valid_parameters(self):
        if self.type == "f":
            for key in [
                "lambda_e",
                "alpha_C",
                "alpha_D",
                "tau_c",
                "tau_d",
            ]:
                if not hasattr(self, key) or getattr(self, key) is None:
                    raise ValueError(f"Parameter {key} is not provided")
        elif self.type == "b":
            for key in [
                "lambda_e",
                "alpha_C",
                "alpha_D",
                "tau_c",
                "tau_d",
            ]:
                if not hasattr(self, key) or getattr(self, key) is None:
                    raise ValueError(f"Parameter {key} is not provided")
        elif self.type == "l":
            for key in [
                "lambda_e",
                "alpha_C",
                "alpha_D",
                "tau_c",
                "tau_d",
            ]:
                if not hasattr(self, key) or getattr(self, key) is None:
                    print(
                        f"Parameter {key} is not provided. If {self.type} is a lateral synapse, it may not be needed if we are dealing with eli or ile synapses"
                    )

        else:
            raise ValueError(f"Unknown synapse type: {self.synapse_type}")


class SynapseSpecs:
    """
    Synapse specs for synapses in a neural network."""

    def __init__(self, model, on_pre, on_post=None, type=None, **params):
        self.model = model
        self.on_pre = on_pre
        self.on_post = on_post
        self.type = type
        self.params = SynapseParameters(type=type, **params)
        self.synapse_objects = {}  # by layer
        self.recent_a = None  # this is dumb lol but works perfectly
        self.recent_e = None

    def create_synapses(
        self,
        layer,
        afferent_group_specs: NeuronSpecs,
        efferent_group_specs: NeuronSpecs,
        target_network=None,
        debug=False,
    ):
        if self.type == "f":
            afferent_group = afferent_group_specs.neuron_groups[layer]
            efferent_group = efferent_group_specs.neuron_groups[layer + 1]

        elif self.type == "b":
            afferent_group = afferent_group_specs.neuron_groups[layer]
            efferent_group = efferent_group_specs.neuron_groups[layer - 1]
        elif self.type == "l":
            afferent_group = afferent_group_specs.neuron_groups[layer]
            efferent_group = efferent_group_specs.neuron_groups[layer]
        afferent_type = afferent_group_specs.neuron_type
        self.recent_a = afferent_type
        efferent_type = efferent_group_specs.neuron_type
        self.recent_e = efferent_type
        if debug:
            print(
                f"Creating synapses from {afferent_group.name} to {efferent_group.name}"
            )
            print(f"afferent type: {afferent_type}, efferent type: {efferent_type}")
            print(f"Model equation for efferent neurons: {efferent_group.equations}")
        synapse_name = f"{afferent_type}{self.type}{efferent_type}_{layer}"
        print(f"*** synapse_name: {synapse_name} ***")
        synapses = Synapses(
            afferent_group,
            efferent_group,
            self.model,
            method="rk4",
            on_pre=self.on_pre,
            on_post=self.on_post,
            name=synapse_name,
        )
        self.synapse_objects[layer] = (
            synapses,
            afferent_group,
            efferent_group,
        )
        target_network.add(synapses)

    def connect_synapses(
        self,
        layer,
        radius,
        avg_no_neurons,
        storage=None,
        data=None,
    ):
        synapses = self.synapse_objects[layer][0]
        afferent_group = self.synapse_objects[layer][1]
        efferent_group = self.synapse_objects[layer][2]
        size_afferent = sqrt(afferent_group.N)
        size_efferent = sqrt(efferent_group.N)
        if data is not None:
            print(
                f"\r *** RETRIEVING DATA FOR {self.recent_a}{self.type}{self.recent_e}_{layer} ***",
                flush=True,
            )
            for j in range(efferent_group.N):
                conv_data = [int(data) for data in data[j]]

                if len(conv_data) == 0:
                    print(f"no connections for neuron {j}")
                else:
                    synapses.connect(i=conv_data, j=j)
        else:
            print(
                f"\r *** GENERATING DATA TO CONNECT synapses from {afferent_group.name} to {efferent_group.name} for layer {layer} ***",
                flush=True,
            )
            scale = size_afferent / size_efferent
            # print(f"for synapses from {afferent_group.name} to {efferent_group.name} scale: {scale}")
            index_list = []  # for debugging
            index_lens = []
            print(f" radius: {radius}")
            for j in range(efferent_group.N):
                row = efferent_group[j].row[0]
                column = efferent_group[j].column[0]
                indexes = self._get_indexes(
                    row,
                    column,
                    size_afferent,
                    scale,
                    radius,
                )
                index_list.append(indexes)
                index_lens.append(len(indexes))

            # the mean number of connections
            mean = np.mean(index_lens)
            print(f"mean: {mean}")
            # probability to get avg_no_neurons connections
            connection_probability = avg_no_neurons / mean
            print(f"connection_probability: {connection_probability}")
            for j in range(efferent_group.N):
                # for each item, in index_list[j] retain with a probability of connection_probability
                indexes = index_list[j]
                indexes = [
                    index
                    for index in indexes
                    if np.random.rand() < connection_probability
                ]
                index_list[j] = indexes
                if len(index_list[j]) == 0:
                    print(f"no connections for neuron {j}")
                    print(type(index_list[j]))
                    print(type(indexes))
                    print(index_list[j])
                else:
                    synapses.connect(i=index_list[j], j=j)
            # flat_indices = []
            # flat_j = []
            # for j, pre_indices in enumerate(index_list):  # indices is a list of lists
            #     flat_indices.extend(pre_indices)  # Append all pre-synaptic indices
            #     flat_j.extend(
            #         [j] * len(pre_indices)
            #     )  # Repeat post-synaptic index for each connection
            # # Convert to NumPy arrays
            # flat_indices = np.array(flat_indices, dtype=int)
            # flat_j = np.array(flat_j, dtype=int)

            # # Connect synapses
            # # if storage is not None:
            # #     print(f"STORING INPUT DATA")
            # #     storage["i_0"] = [int(index) for index in flat_indices]
            # #     storage["j_0"] = [int(index) for index in flat_j]

            # synapses.connect(i=flat_indices, j=flat_j)

            if storage is not None:
                print(
                    f"Storing indexes for {self.recent_a}{self.type}{self.recent_e}_{layer}"
                )
                converted_index_list = []
                for j in range(efferent_group.N):
                    converted_index_list.append([int(index) for index in index_list[j]])
                storage[f"{self.recent_a}{self.type}{self.recent_e}_{layer}"] = (
                    converted_index_list
                )
                print(
                    f"Stored indexes for {self.recent_a}{self.type}{self.recent_e}_{layer}"
                )

            # I want the variance and average of indexes
            print(f"mean: {np.mean([len(indexes) for indexes in index_list])}")
            print(f"variance: {np.var([len(indexes) for indexes in index_list])}")
        self._set_synapse_parameters(synapses)
        if self.recent_a == self.recent_e:
            synapses.w = "rand()"
            synapses.delay = "1*ms + 9*ms*rand()"
            synapses.plasticity = 1
        else:
            synapses.w = 1
            synapses.delay = 0.1 * ms

    # Set parameters after synapses are connected
    def _set_synapse_parameters(self, synapses):
        print("Setting synapse parameters")
        safe_values = [
            "lambda_e",
            "lambda_i",
            "A_plus",
            "alpha_C",
            "alpha_D",
            "tau_c",
            "tau_d",
            "learning_rate",
        ]
        excluded_values = []
        set_values = []
        for param in safe_values:
            try:
                # print(f"Setting {param} to {getattr(self.params, param)}")
                setattr(synapses, param, getattr(self.params, param))
                set_values.append(param)
            except Exception as e:
                excluded_values.append(param)
                pass
        print(f"*** Set values: {set_values} ***")
        print(f"*** Excluded values: {excluded_values} ***")

    def _get_indexes(self, row, col, size_efferent, scale, radius):
        # This is where the neuron in the post layer is centred in the previous layer
        col_centre = int(scale * col + scale / 2)

        # This is where the neuron in the post layer is centred in the previous layer
        row_centre = int(scale * row + scale / 2)

        # Define min and max values for the row and column to reduce computational load
        col_min = max(0, col_centre - radius - 3)
        col_max = min(size_efferent - 1, col_centre + radius + 3)
        row_min = max(0, row_centre - radius - 3)
        row_max = min(
            size_efferent - 1, row_centre + radius + 3
        )  # If 3 feels random it kinda is - just guessing it's good as it's 2 (max scale) + 1 so no cheeky stuff
        # print(
        #     f" row range: {row_min} - {row_max}; col range: {col_min} - {col_max}; size: {(row_min - row_max) *(col_min - col_max)}"
        # )
        # Create the row and column ranges
        row_range = np.arange(row_min, row_max)
        col_range = np.arange(col_min, col_max)
        # print(f"row_min:{row_min}")
        # print(f"row_max:{row_max}")
        # Create the row and column coordinates
        row_coords = np.repeat(row_range, len(col_range))
        col_coords = np.tile(col_range, len(row_range))

        accepted_rows = np.array([])
        accepted_columns = np.array([])

        for col, row in zip(col_coords, row_coords):
            if np.sqrt((col - col_centre) ** 2 + (row - row_centre) ** 2) < radius:
                accepted_rows = np.append(accepted_rows, row)
                accepted_columns = np.append(accepted_columns, col)
        # print(f"num of connections: {len(accepted_rows)}")
        indexes = (accepted_rows * size_efferent + accepted_columns).astype(int)
        return indexes
