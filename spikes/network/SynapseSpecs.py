from brian2 import *
import time
from typing import Union

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

from .NeuronSpecs import NeuronSpecs
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
        
        # print(f"*** synapse_name: {synapse_name} ***")
        # print(f"model for {synapse_name} at creation of synapses:")
        # print(self.neuron_model)
        # print(f"on_pre for {synapse_name} at creation of synapses:")
        # print(self.pre_point)
        # print(f"on_post for {synapse_name} at creation of synapses:")
        # print(self.post_point)
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
        layer: int,
        radius: float,
        avg_no_neurons: float,
        storage: Union["load", "save"] = None,
        storage_path: str = None,
    ) -> None:
        synapses, afferent_group, efferent_group = self.synapse_objects[layer]

        # 1) compute map sizes and scale
        size_aff = int(np.sqrt(afferent_group.N))
        size_eff = int(np.sqrt(efferent_group.N))
        scale    = size_aff / size_eff

        print(f"\r*** Connecting {afferent_group.name}→{efferent_group.name} (layer {layer}), "
              f"radius={radius}, scale={scale} ***", flush=True)

        # 2) try loading
        if storage == "load":
            try:
                print(storage_path)
                data = np.load(os.path.join(
                    storage_path,
                    f"{layer}_{afferent_group.name}_{efferent_group.name}_synapses.npz"
                ))
                print("Loading synapses from file")
                i=data["arr_1"]
                j=data["arr_2"]
                print("connecting synapses")
                synapses.connect(i=i, j=j)
                print("setting weights and delays")
                w     = data["arr_0"]
                print(w)
                synapses.w = w
                delay = data["arr_3"]
                print(delay)
                synapses.delay = delay * msecond
                print("setting synapse parameters")
                self._set_synapse_parameters(synapses)
                if self.recent_a == self.recent_e:
                    synapses.plasticity = 1
                return
            except Exception as e:
                print(f"Error loading synapses: {e}")
                print("Load failed, regenerating…")

        # 3) precompute the offsets & centers once
        print("Precomputing offsets and centers")
        self._prepare_receptive_field(size_aff, scale, radius)
        # 4) VECTORISED receptive‐field + connect
        N_e = efferent_group.N
        print("precomputed")
        # 4a) gather all rows & cols (0-based) in one shot
        rows = efferent_group.row[:].astype(int)
        cols = efferent_group.column[:].astype(int)
        print("gathered rows and cols")
        # 4b) lookup each neuron’s centre in afferent coords
        crs = self._center_rows[rows]
        ccs = self._center_cols[cols]
        print("broadcasting")
        # 4c) broadcast the circular offsets
        all_rs = crs[:, None] + self._dr_offsets[None, :]
        all_cs = ccs[:, None] + self._dc_offsets[None, :]
        print("applying clippings")
        # 4d) apply the same exclusive-window clipping
        R      = self._rf_radius + self._rf_pad
        rmins  = np.clip(crs - R, 0, size_aff - 1)[:, None]
        rmaxs  = np.clip(crs + R, 0, size_aff - 1)[:, None]
        cmins  = np.clip(ccs - R, 0, size_aff - 1)[:, None]
        cmaxs  = np.clip(ccs + R, 0, size_aff - 1)[:, None]
        window = (
            (all_rs >= rmins) & (all_rs < rmaxs) &
            (all_cs >= cmins) & (all_cs < cmaxs)
        )
        print("flattening")
        # 4e) flatten into two 1D arrays for i,j
        flats = all_rs * size_aff + all_cs     # shape (N_e, M)
        i_inds = flats[window]                  # all source indices
        counts = window.sum(axis=1)             # sources per neuron
        j_inds = np.repeat(np.arange(N_e), counts)
        print("probabilistically culling")
        # 5) decide which to keep
        max_conn = counts.max()
        print("max_conn", max_conn)
        p = avg_no_neurons / max_conn
        print("p", p)
        keep = np.random.rand(i_inds.size) < p
        print("keeping", keep.sum(), "connections")
        # 6) one‐shot connect
        if keep.any():
            synapses.connect(i=i_inds[keep], j=j_inds[keep])
        else:
            print("Warning: no connections kept!")

        # 7) set your STDP/non‐STDP parameters and save if needed
        self._set_synapse_parameters(synapses)
        if storage == "save":
            os.makedirs(os.path.join(storage_path, "epoch_0"), exist_ok=True)
            np.savez(
                os.path.join(storage_path, "epoch_0",
                             f"{layer}_{afferent_group.name}_{efferent_group.name}_synapses.npz"),
                synapses.w, synapses.i, synapses.j, synapses.delay
            )

    def _prepare_receptive_field(self,
                                size_afferent: int,
                                scale: float,
                                radius: float,
                                padding: int = 3) -> None:
        """
        Precompute the circular offsets and center‐lookup tables
        so that each call to _get_indexes_precomputed is as cheap as
        a few adds and one boolean mask.
        """
        self._rf_size   = int(size_afferent)
        self._rf_scale  = scale
        self._rf_radius = radius
        self._rf_pad    = padding

        # radius+padding defines half‐width of the square window
        R = radius + padding

        # build a full square of relative offsets [-R .. R)
        drs, dcs = np.meshgrid(
            np.arange(-R, R),
            np.arange(-R, R),
            indexing='ij'
        )

        # mask to only keep the circle of true radius
        circle_mask = (drs**2 + dcs**2) <= radius**2
        self._dr_offsets = drs[circle_mask]
        self._dc_offsets = dcs[circle_mask]

        # how many rows/cols in the efferent grid?
        n = int(self._rf_size / self._rf_scale)   # → 64/1 = 64
        centers = (scale * np.arange(n) + scale/2).astype(int)
        self._center_rows = centers
        self._center_cols = centers


    def _get_indexes_precomputed(self, row: int, col: int) -> np.ndarray:
        """
        Super‐fast lookup of flat indices using precomputed offsets.
        Must call _prepare_receptive_field first.
        """
        cr = self._center_rows[row]
        cc = self._center_cols[col]

        # absolute candidate coords
        rs = cr + self._dr_offsets
        cs = cc + self._dc_offsets

        # clip to the same exclusive window the original used:
        R = self._rf_radius + self._rf_pad
        rmin = max(0, cr - R)
        cmin = max(0, cc - R)

        in_win = (
            (rs >= rmin) & (rs <  min(self._rf_size-1, cr + R)) &
            (cs >= cmin) & (cs <  min(self._rf_size-1, cc + R))
        )

        rs = rs[in_win]
        cs = cs[in_win]
        return (rs * self._rf_size + cs).astype(int)


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
