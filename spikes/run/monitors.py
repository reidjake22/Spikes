<<<<<<< HEAD
"""
Module: Monitor Utilities for Spiking Neural Networks

This module provides a set of tools for creating, managing, and visualizing
various types of monitors in spiking neural network simulations using Brian2.

TODO: SEND EXPLICIT VISUALISATIONS TO VISUALISATION MODULE
Classes:
    Monitors: A utility class for creating and managing monitors for
              neuron groups in a spiking neural network.
"""

from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
=======
>>>>>>> jakes_working_repo
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from brian2 import *


<<<<<<< HEAD
class Monitors:
    """
    A class to manage and visualize monitors in spiking neural network simulations.

    Attributes:
        network (brian2.Network): The Brian2 network object to which monitors are added.
        n_layers (int): Number of layers in the network.
        monitors (dict): Dictionary to store monitors, keyed by (layer_name, monitor_type).
        monitor_data (dict): Dictionary to store processed monitor data.
    """

    def __init__(self, network, n_layers):
        """
        Initialize the Monitors class.

        Args:
            network (brian2.Network): The Brian2 network object.
            n_layers (int): Number of layers in the network.
        """
=======
class MonitorManager:
    """
    A class to manage, toggle, and store data from monitors in a Brian2 simulation.
    Allows for visualization of spiking activity and voltage dynamics.
    """

    def __init__(self, network):
        """Initialize the MonitorManager."""
>>>>>>> jakes_working_repo
        self.network = network
        self.monitors = {}  # Active monitors: {(layer, monitor_type, excitatory): monitor}
        self.saved_data = {}  # Stores removed monitor data

<<<<<<< HEAD
    def create_monitor(self, neuron_group, monitor_type, layer, **kwargs):
        """
        Create a monitor for a given neuron group.

        Args:
            neuron_group (brian2.NeuronGroup): The neuron group to monitor.
            monitor_type (str): Type of monitor (e.g., 'spike', 'voltage').
            layer (int): Layer number associated with the neuron group.
            **kwargs: Additional arguments for the monitor constructor.

        Returns:
            brian2.Monitor: The created monitor.
        """
=======
    def create_monitor(self, monitor_type: str, layer: int, exc: bool, **kwargs):
        """
        Create and add a monitor for a neuron group in the network.

        Args:
            monitor_type (str): Type of monitor ("spike" or "voltage").
            layer (int): The layer number.
            exc (bool): Whether the group is excitatory.
        """
        if (layer, monitor_type) in self.monitors:
            raise ValueError(
                f"Monitor for layer {layer} and type {monitor_type} already exists."
            )

        # Find the neuron group in the network
        if layer == 0:
            group_name = "p_0"
        elif exc:
            group_name = f"e_{layer}"
        else:
            group_name = f"i_{layer}"
        neuron_group = next(
            (
                obj
                for obj in self.network.objects
                if hasattr(obj, "name") and obj.name == group_name
            ),
            None,
        )

        if neuron_group is None:
            raise ValueError(f"Neuron group '{group_name}' not found in the network.")

        # Monitor constructors
>>>>>>> jakes_working_repo
        constructors = {
            "spike": SpikeMonitor,
            "voltage": lambda group, **kw: StateMonitor(
                group, variables="v", record=True, **kw
            ),
        }

        if monitor_type not in constructors:
            raise ValueError(f"Unsupported monitor type: {monitor_type}")

        monitor = constructors[monitor_type](neuron_group, **kwargs)
        self.network.add(monitor)
        self.monitors[(layer, monitor_type, exc)] = monitor
        return monitor

<<<<<<< HEAD
    def setup_excitatory_monitors(self, layers, monitor_type, **kwargs):
        """
        Setup monitors for excitatory neuron groups across layers.

        Args:
            layers (list[int]): List of layer indices to set up monitors for.
            monitor_type (str): Type of monitor to set up.
            **kwargs: Additional arguments for monitor setup.
        """
        for layer in layers:
            layer_name = f"e_{layer}" if layer != 0 else "p_0"
            group = next(
                (
                    obj
                    for obj in self.network.objects
                    if hasattr(obj, "name") and obj.name == layer_name
                ),
                None,
            )
            if group is None:
                raise ValueError(f"Neuron group '{layer_name}' not found.")
            self.create_monitor(group, monitor_type, layer, **kwargs)

    def setup_poisson_monitors(self, monitor_type):
        """
        Setup monitors for the Poisson input layer.

        Args:
            monitor_type (str): Type of monitor to set up.
        """
        layer_name = "p_0"
        group = next(
            (
                obj
                for obj in self.network.objects
                if hasattr(obj, "name") and obj.name == layer_name
            ),
            None,
        )
        if group is None:
            raise ValueError(f"Neuron group '{layer_name}' not found.")
        self.create_monitor(group, monitor_type, 0)

    def toggle_monitoring(self, layer_number=None, monitor_type=None, enable=True):
        """
        Toggle monitoring for specified layers and monitor types.

        Args:
            layer_number (int, optional): Layer number to toggle monitoring for.
                                          Defaults to None (all layers).
            monitor_type (str, optional): Type of monitor to toggle. Defaults to None (all types).
            enable (bool): Enable or disable monitoring. Defaults to True.

        Returns:
            str: A message indicating the status of the toggled monitors.
=======
    def remove_monitor(self, layer, monitor_type, exc):
        """
        Remove a monitor while preserving its data.

        Args:
            layer (int): The layer number.
            monitor_type (str): The type of monitor ("spike" or "voltage").
            exc (bool): Whether the group is excitatory.
>>>>>>> jakes_working_repo
        """
        identifier = (layer, monitor_type, exc)
        monitor = self.monitors.pop(identifier, None)
        if monitor is None:
            raise ValueError(
                f"No monitor of type {monitor_type, exc} found in layer {layer}."
            )

<<<<<<< HEAD
    def get_monitors(self, layer_number=None, monitor_type=None):
        """
        Retrieve monitors based on specified criteria.

        Args:
            layer_number (int, optional): Layer number to retrieve monitors for. Defaults to None (all layers).
            monitor_type (str, optional): Type of monitor to retrieve. Defaults to None (all types).

        Returns:
            list: A list of monitors matching the criteria.
        """
        filtered_monitors = []
        criteria = lambda k: (
            (layer_number is None or k[0] == layer_number)
            and (monitor_type is None or k[1] == monitor_type)
        )

        for key, monitor in self.monitors.items():
            if criteria(key):
                filtered_monitors.append(monitor)

        return filtered_monitors

    def visualise_monitor(self, layer_number, monitor_type):
        """
        Visualize monitor data for a specified layer and monitor type.

        Args:
            layer_number (int): The layer number to visualize.
            monitor_type (str): The type of monitor to visualize ('spike', 'voltage', etc.).

        Returns:
            str: A message indicating the status of the visualization or an error message
                 if no matching monitors are found.
        """
        monitors = self.get_monitors(layer_number, monitor_type)
        if not monitors:
            return "No monitors matched the criteria."
        else:
            monitor = monitors[0]

=======
        # Store data before removing
>>>>>>> jakes_working_repo
        if monitor_type == "spike":
            self.saved_data[identifier] = monitor.spike_trains()
        elif monitor_type == "voltage":
            self.saved_data[identifier] = {
                "times": monitor.t[:],
                "voltages": {i: monitor.v[i][:] for i in range(len(monitor.record))},
            }

<<<<<<< HEAD
    def bin_spikes(self, monitor, num_stimuli, length_stimuli):
        """
        Bin spike data into histograms based on the stimuli and their durations.

        Args:
            monitor (brian2.SpikeMonitor): The spike monitor containing spike trains.
            num_stimuli (int): The number of stimuli to bin spikes for.
            length_stimuli (float): The duration of each stimulus in simulation time.

        Returns:
            numpy.ndarray: A 2D array where rows represent neurons and columns represent stimulus bins.
        """
        spikes = monitor.spike_trains()
        num_neurons = monitor.source.N
=======
        self.network.remove(monitor)

    def get_monitor_data(self, layer, monitor_type, exc):
        """Retrieve data from active or deleted monitors."""
        identifier = (layer, monitor_type, exc)
        monitor = self.monitors.get(identifier, None)
        if monitor is not None:
            return monitor
        if identifier in self.saved_data:
            return self.saved_data[identifier]
        raise ValueError(
            f"No data found for layer {layer}, monitor type {monitor_type}."
        )

    def bin_spikes(self, layer, exc, num_stimuli, length_stimuli):
        """Bin spike data into histograms based on stimuli and durations."""
        monitor = self.get_monitor_data(layer, "spike", exc)
        if isinstance(monitor, dict):  # Stored spike train data
            spikes = monitor
        else:
            spikes = monitor.spike_trains()

        num_neurons = len(spikes)
>>>>>>> jakes_working_repo
        store = np.zeros((num_neurons, num_stimuli))
        bins = np.arange(0, length_stimuli * (num_stimuli + 1), length_stimuli)
<<<<<<< HEAD
        for key, value in spikes.items():
            counts, edges = np.histogram(value, bins=bins)
            store[key] = counts
        # Ideally would happen here but layer is inaccessible
        return store

    def plot_spikes(self, layer, type, index, num_stimuli, length_stimuli):
        """
        Plot the histogram of spike counts for a specified neuron and stimulus bins.

        Args:
            layer (int): The layer number.
            type (str): The type of monitor ('spike').
            index (int): The index of the neuron to plot.
            num_stimuli (int): The number of stimuli.
            length_stimuli (float): The duration of each stimulus in simulation time.
        """
        monitor = self.get_monitors(layer, type)[0]
=======

        for neuron_idx, spike_times in spikes.items():
            counts, _ = np.histogram(spike_times, bins=bins)
            store[neuron_idx] = counts

        return store

    def bin_voltages(self, layer, exc, num_stimuli, length_stimuli):
        """Bin voltage data into average values per neuron and stimulus."""
        monitor = self.get_monitor_data(layer, "voltage", exc)
>>>>>>> jakes_working_repo

        if isinstance(monitor, dict):  # Stored voltage data
            times, voltages = monitor["times"], monitor["voltages"]
        else:
            times, voltages = monitor.t, {
                i: monitor.v[i] for i in range(len(monitor.record))
            }

        num_neurons = len(voltages)
        store = np.zeros((num_neurons, num_stimuli))
        bins = np.arange(0, length_stimuli * (num_stimuli + 1), length_stimuli)

        for neuron_idx in range(num_neurons):
            digitized = np.digitize(times, bins) - 1  # Assign times to bins
            for i in range(num_stimuli):
                store[neuron_idx, i] = np.mean(voltages[neuron_idx][digitized == i])

<<<<<<< HEAD
    def generate_spike_heatmap(
        self, layer, type, num_stimuli, length_stimuli, layer_length, stimulus_index
    ):
        """
        Generate a heatmap of spike activity for a specified stimulus.

        Args:
            layer (int): The layer number.
            type (str): The type of monitor ('spike').
            num_stimuli (int): The number of stimuli.
            length_stimuli (float): The duration of each stimulus in simulation time.
            layer_length (int): The side length of the layer grid.
            stimulus_index (int): The index of the stimulus to generate the heatmap for.

        Returns:
            numpy.ndarray: A 2D array representing spike counts as a heatmap.
        """
        monitor = self.get_monitors(layer, type)[0]
        if (layer, "spike", "histogram") not in self.monitor_data:
            store = self.bin_spikes(monitor, num_stimuli, length_stimuli)
            self.monitor_data[(layer, "spike", "histogram")] = store
        else:
            store = self.monitor_data[(layer, "spike", "histogram")]
        heatmap = store[:, stimulus_index].reshape(layer_length, layer_length)
        return heatmap

    def display_spike_heatmap(
        self, layer, type, num_stimuli, length_stimuli, layer_length, stimulus_index
    ):
        """
        Display a heatmap of spike activity for a specified stimulus.

        Args:
            layer (int): The layer number.
            type (str): The type of monitor ('spike').
            num_stimuli (int): The number of stimuli.
            length_stimuli (float): The duration of each stimulus in simulation time.
            layer_length (int): The side length of the layer grid.
            stimulus_index (int): The index of the stimulus to display the heatmap for.
        """
        heatmap = self.generate_spike_heatmap(
            layer, type, num_stimuli, length_stimuli, layer_length, stimulus_index
        )
        plt.imshow(heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.show()

    def animate_spike_heatmap(
        self, layer, type, num_stimuli, length_stimuli, layer_length
    ):
        """
        Animate a sequence of heatmaps representing spike activity across stimuli.

        Args:
            layer (int): The layer number.
            type (str): The type of monitor ('spike').
            num_stimuli (int): The number of stimuli.
            length_stimuli (float): The duration of each stimulus in simulation time.
            layer_length (int): The side length of the layer grid.
        """

        fig, ax = plt.subplots()
        heatmap = self.generate_spike_heatmap(
            layer, type, num_stimuli, length_stimuli, layer_length, 0
        )
        im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax)  # Add color bar
        ax.set_title(f"Spike Heatmap - layer {layer}")  # Initial title
=======
        return store

    def plot_raster(self, layer, exc, index_range=None):
        """Generate a raster plot of spike events."""
        spikes = self.get_monitor_data(layer, "spike", exc)
        plt.figure(figsize=(10, 6))

        for neuron_idx, spike_times in spikes.items():
            if index_range and (
                neuron_idx < index_range[0] or neuron_idx > index_range[1]
            ):
                continue
            plt.scatter(
                spike_times,
                np.full_like(spike_times, neuron_idx),
                marker="|",
                color="black",
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title(f"Raster Plot - Layer {layer}")
        plt.show()

    def plot_spike_histogram(self, layer, exc, num_stimuli, length_stimuli, animate=False):
        """Plot histogram of spike rates per neuron per stimulus."""
        spike_data = self.bin_spikes(layer, exc, num_stimuli, length_stimuli)

        if spike_data.shape[0] == 0:
            raise ValueError(f"No neurons recorded in layer {layer} for spikes.")

        fig, ax = plt.subplots(figsize=(10, 6))
>>>>>>> jakes_working_repo

        def update(i):
            ax.clear()
            ax.hist(spike_data[:, i], bins=20, color="blue", alpha=0.7)
            ax.set_title(f"Spike Rate Histogram - Stimulus {i}")
            ax.set_xlabel("Spike Count")
            ax.set_ylabel("Neuron Frequency")

        if animate:
            ani = animation.FuncAnimation(fig, update, frames=num_stimuli, repeat=True)
            plt.show()
        else:
            for i in range(num_stimuli):
                update(i)
                plt.pause(0.5)

    def plot_voltage_histogram(self, layer, exc, num_stimuli, length_stimuli, animate=True):
        """Plot histogram of voltage values per neuron per stimulus."""
        voltage_data = self.bin_voltages(layer, exc, num_stimuli, length_stimuli)

        if voltage_data.shape[0] == 0:
            raise ValueError(f"No neurons recorded in layer {layer} for voltage.")

        fig, ax = plt.subplots(figsize=(10, 6))

        def update(i):
            ax.clear()
            ax.hist(voltage_data[:, i], bins=20, color="red", alpha=0.7)
            ax.set_title(f"Voltage Histogram - Stimulus {i}")
            ax.set_xlabel("Voltage (mV)")
            ax.set_ylabel("Neuron Frequency")

        if animate:
            ani = animation.FuncAnimation(fig, update, frames=num_stimuli, repeat=False)
            plt.show()
        else:
            for i in range(num_stimuli):
                update(i)
                plt.pause(0.5)

def extract_spike_heatmap(spike_monitor, width, n_filters=1, is_input=False):
    """
    Extracts spike data and converts it into a heatmap-ready format.

    Parameters:
    - spike_monitor: Brian2 SpikeMonitor object
    - width: Grid width (e.g., 128 for e1, 64 for e2/e3, 32 for inhibitory)
    - n_filters: Number of filters (default = 1 for non-input neurons)
    - is_input: Boolean flag to reshape into (width, width, n_filters)

    Returns:
    - A 2D (or 3D for input neurons) numpy array of spike counts
    """
    print(f"Extracting spike data for {spike_monitor.source.name}...")
    neuron_quantity = width * width * n_filters
    spike_list = [[] for _ in range(neuron_quantity)]

    for t, i in zip(spike_monitor.t, spike_monitor.i):
        spike_list[i].append(t)

    spike_counts = np.array([len(spike_list[i]) for i in range(neuron_quantity)])
    return spike_counts.reshape(width, width, n_filters) if is_input else spike_counts.reshape(width, width)

def extract_binned_spike_heatmap(spike_monitor, bin_size, start_time, end_time, width, n_filters=1, is_input=False):
    """
    Bins spike data within a time window and creates a heatmap.

    Parameters:
    - spike_monitor: Brian2 SpikeMonitor object
    - bin_size: Time window per bin (in seconds)
    - start_time: Start time of binning
    - end_time: End time of binning
    - width: Grid width (e.g., 128 for e1, 64 for e2/e3, 32 for inhibitory)
    - n_filters: Number of filters (default = 1)

    Returns:
    - A 4D numpy array (num_bins, width, width, n_filters) of binned spike counts
    """
    num_bins = int((end_time - start_time) / bin_size)
    binned_spike_counts = np.zeros((num_bins, width, width, n_filters))

    for t, i in zip(spike_monitor.t, spike_monitor.i):
        if start_time <= t < end_time:
            bin_idx = int((t - start_time) / bin_size)

            # Extract filter index, row, and column
            row, remainder = divmod(i, width * n_filters)
            col, f_idx = divmod(remainder, n_filters)

            # Bounds check to avoid indexing errors
            if f_idx < n_filters and row < width and col < width:
                binned_spike_counts[bin_idx, row, col, f_idx] += 1
            else:
                print(f"Warning: Neuron index {i} (row={row}, col={col}, f_idx={f_idx}) is out of bounds.")

    return binned_spike_counts

def test_extract_binned_spike_heatmap(spike_monitor, bin_size, start_time, end_time, width, n_filters=1, is_input=False):
    num_bins = int((end_time - start_time) / bin_size)
    spike_counts = [[] for _ in range(width * width * n_filters)]
    for t,i in zip(spike_monitor.t, spike_monitor.i):
        spike_counts[i].append(t)
    binned_spike_counts = np.zeros((num_bins, width, width, n_filters))
    for i in range(width):
        for j in range(width):
            for k in range(n_filters):
                for t in spike_counts[i*k*j + j*k + k]:
                    bin_idx = int((t - start_time) // bin_size)
                    if bin_idx < num_bins:
                        binned_spike_counts[bin_idx, i, j, k] += 1
                    else:
                        print("something weird!")
    return binned_spike_counts

def display_spike_heatmap(spike_heatmap, filename, is_input=False):
    """
    Displays and saves a heatmap of spike counts.

    Parameters:
    - spike_heatmap: 2D or 3D numpy array of spike counts
    - filename: Output file name for saving the heatmap
    - is_input: Boolean flag for input neurons (3D heatmap)

    Saves:
    - A heatmap image
    """
    if is_input:
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))  # Create 2x4 grid of heatmaps
        for filter_idx, ax in enumerate(axs.flatten()):
            im = ax.imshow(spike_heatmap[:, :, filter_idx], cmap='hot', aspect='auto')
            ax.set_title(f"Filter {filter_idx}")
            ax.axis('off')
        fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02, pad=0.04)
        plt.tight_layout()

    else:
        fig = plt.figure(figsize=(8, 6))  # No need for plt.subplots()
        im = plt.imshow(spike_heatmap, cmap='hot', aspect='auto')
        plt.colorbar(im)
        plt.title("Spike Heatmap")

    plt.savefig(filename)
    plt.show()

# Modify the function to support both single-filter and multi-filter cases
def plot_binned_heatmaps(binned_data, filename, title_prefix):
    """
    Plots and saves heatmaps for binned spike data.

    Parameters:
    - binned_data: 3D or 4D numpy array (num_bins, width, width) or (num_bins, width, width, n_filters)
    - filename: Output file name for saving the heatmap
    - title_prefix: Title prefix for labeling bins

    Saves:
    - A series of heatmaps over time bins and filters
    """
    num_bins = binned_data.shape[0]
    width = binned_data.shape[1]
    n_filters = 1 if len(binned_data.shape) == 3 else binned_data.shape[3]
    if n_filters > 1:
        fig, axs = plt.subplots(num_bins, n_filters, figsize=(n_filters * 4, num_bins * 4))

        # Ensure axs is always a 2D array for iteration, even when num_bins or n_filters is 1
        if num_bins == 1:
            axs = axs[np.newaxis, :]
        for bin_idx in range(num_bins):
            for filter_idx in range(n_filters):
                heatmap = binned_data[bin_idx, :, :, filter_idx]
                im = axs[bin_idx, filter_idx].imshow(heatmap, cmap="hot", aspect="equal")
                axs[bin_idx, filter_idx].set_title(f"{title_prefix} Bin {bin_idx + 1} - Filter {filter_idx + 1}")
                axs[bin_idx, filter_idx].axis("off")
        fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    else:
        cols = math.ceil(math.sqrt(num_bins))
        rows = math.ceil(num_bins / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if num_bins == 1:
            axs = axs[np.newaxis, :]
        for bin_idx in range(num_bins):
            heatmap = binned_data[bin_idx, :, :]
            curr_row, curr_col = divmod(bin_idx, cols)
            im = axs[curr_row, curr_col].imshow(heatmap, cmap="hot", aspect="equal")
            axs[curr_row, curr_col].set_title(f"{title_prefix} Bin {bin_idx + 1}")
            axs[curr_row, curr_col].axis("off")
        fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def the_full_monty(self, directory, filename):
        """
        Save all monitor data to a file.

        Args:
            directory (str): The directory to save the data.
            filename (str): The filename to save the data as.

        Returns:
            dict: A dictionary containing the spike trains data.
        """
        import os

        if not os.path.exists(directory):
            os.makedirs(directory)
        data = {}
        for key, item in self.monitors.items():
            if key[1] == "spike":
                spikes = item.spike_trains()
                data[str(key[0])] = spikes

        np.savez(f"{directory}/{filename}.npz", **data)
        print("Data saved successfully.")
        return data

    def bin_poisson_spikes(self, num_stimuli, length_stimuli):
        """
        Bin spike data for the Poisson input layer into histograms.

        Args:
            num_stimuli (int): The number of stimuli.
            length_stimuli (float): The duration of each stimulus in simulation time.

        Returns:
            numpy.ndarray: A 2D array where rows represent neurons and columns represent stimulus bins.
        """
        monitor = self.get_monitors(0, "spike")[0]
        spikes = monitor.spike_trains()
        num_neurons = monitor.source.N
        store = np.zeros((num_neurons, num_stimuli))
        edges = 0
        bins = np.arange(0, length_stimuli * (num_stimuli + 1), length_stimuli)
        for key, value in spikes.items():
            counts, edges = np.histogram(value, bins=bins)
            store[key] = counts
        return store

    def return_spike_data(self):
        """
        Return the spike data for all monitors.

        Returns:
            dict: A dictionary containing the spike trains data.
        """
        data = {}
        for key, item in self.monitors.items():
            if key[1] == "spike":
                spikes = item.spike_trains()
                data[str(key[0])] = spikes
        return data
