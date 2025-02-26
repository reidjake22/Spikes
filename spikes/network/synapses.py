from brian2 import *
import time

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
import pickle
import os


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
            missing = []
            for key in [
                "lambda_e",
                "alpha_C",
                "alpha_D",
                "tau_c",
                "tau_d",
            ]:
                if not hasattr(self, key) or getattr(self, key) is None:
                    missing.append(key)
            if not missing == []:
                print(f"Missing parameters unless eli or ile: {missing}")

        else:
            raise ValueError(f"Unknown synapse type: {self.synapse_type}")


class SynapseSpecs:
    """
    Synapse specs for synapses in a neural network."""

    def __init__(self, model, on_pre, on_post=None, type=None, name=None, **params):
        self.neuron_model = model
        self.pre_point = on_pre
        self.post_point = on_post
        self.type = type
        self.name = name

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
        synapse_name = f"{afferent_type}{self.type}{efferent_type}_{layer}"
        
        print(f"*** synapse_name: {synapse_name} ***")
        print(f"model for {synapse_name} at creation of synapses:")
        print(self.neuron_model)
        print(f"on_pre for {synapse_name} at creation of synapses:")
        print(self.pre_point)
        print(f"on_post for {synapse_name} at creation of synapses:")
        print(self.post_point)
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model=self.neuron_model,
            method="rk4",
            on_pre=self.pre_point,
            on_post=self.post_point,
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
    ):
        synapses = self.synapse_objects[layer][0]
        afferent_group = self.synapse_objects[layer][1]
        efferent_group = self.synapse_objects[layer][2]
        size_afferent = sqrt(afferent_group.N)
        size_efferent = sqrt(efferent_group.N)

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

        mean = np.max(index_lens)  # was np.mean(index_lens)
        print(f"mean: {mean}")
        # probability to get avg_no_neurons connections
        connection_probability = avg_no_neurons / mean
        time.sleep(5)
        print(f"neuron no. {j}")
        print(f"connection_probability: {connection_probability}")
        new_index_list = []
        for j in range(efferent_group.N):
            # for each item, in index_list[j] retain with a probability of connection_probability
            indexes = index_list[j]
            new_indexes = [
                index for index in indexes if np.random.rand() < connection_probability
            ]

            new_index_list.append(new_indexes)
            if len(new_index_list[j]) == 0:
                print(f"no connections for neuron {j}")
            else:
                synapses.connect(i=new_index_list[j], j=j)
        print(
            f"the mean difference between original indexes and new indexes is {np.mean([len(indexes) - len(new_indexes) for indexes, new_indexes in zip(index_list, new_index_list)])}"
        )
        # I want the variance and average of indexes
        print(f"mean: {np.mean([len(indexes) for indexes in index_list])}")

        # Create a directory for storing the data if it doesn't exist
        directory = "simulation_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the file path
        file_path = os.path.join(directory, f"synapse_data_layer_{afferent_group.name}_{efferent_group.name}_{layer}.pkl")

        # Store the index list and new index list
        with open(file_path, "wb") as file:
            pickle.dump({"index_list": index_list, "new_index_list": new_index_list}, file)
        
        print(f"Data stored in {file_path}")
        time.sleep(5)

        # print(f"variance: {np.var([len(indexes) for indexes in index_list])}")
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
            if (
                np.sqrt((col - col_centre) ** 2 + (row - row_centre) ** 2) <= radius
            ):  # HAVE CHANGED THIS!
                accepted_rows = np.append(accepted_rows, row)
                accepted_columns = np.append(accepted_columns, col)
        indexes = (accepted_rows * size_efferent + accepted_columns).astype(int)
        return indexes

class SynapseSpecsInfo:
    def __init__(self, synapse_specs):
        # I want to save data on weights and connectivity for all synapses.
        synapse_groups = synapse_specs.synapse_objects
        self.synapse_info = {}
        for key, synapse in synapse_groups.items():
            self.synapse_info[synapse[0].name] = {
                "source": synapse[0].source.name,
                "target": synapse[0].target.name,
                "i": synapse[0].i,
                "j": synapse[0].j,
                "w": synapse[0].w,
                "d": synapse[0].delay,
            }