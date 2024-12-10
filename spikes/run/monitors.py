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
import numpy as np

# class TimeAvgVoltageMonitor:
#     def __init__(self, monitor_group):
#         self.monitor_group = monitor_group
#         self.voltage_averager = NeuronGroup(
#             N=self.monitor_group.N,
#             model="""total_charge: coulomb
#             average_voltage = total_charge / t : volt""",
#         )

#         vm_totaler = Synapses(
#             self.monitor_group,
#             self.voltage_averager,
#             "total_charge_post = total_charge_post + v_pre * dt : coulomb",
#         )

#         vm_totaler.connect("i == j")
#         self.vm_monitor = StateMonitor(self.voltage_averager, "average_voltage", record=True)


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
        self.network = network
        self.n_layers = n_layers
        self.monitors = {}  # Dictionary to store monitors by (layer_name, monitor_type)
        self.monitor_data = (
            {}
        )  # Dictionary to store monitor data by (layer_name, monitor_type, data_type)

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
        constructors = {
            "spike": SpikeMonitor,
            "voltage": lambda group, **kw: StateMonitor(
                group, variables="v", record=True, **kw
            ),
            "pop_avg_spike": PopulationRateMonitor,
            # "time_avg_spike": PopulationRateMonitor,
            # "time_avg_voltage": TimeAvgVoltageMonitor,
        }
        if monitor_type not in constructors:
            raise ValueError(f"Unsupported monitor type: {monitor_type}")

        monitor = constructors[monitor_type](
            neuron_group, name=f"{monitor_type}__{neuron_group.name}", **kwargs
        )
        # monitor.name = f"{monitor_type}__{neuron_group.name}"
        self.network.add(monitor)
        self.monitors[(layer, monitor_type)] = monitor
        return monitor

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
        """
        toggled = []
        # Define criteria for toggling
        criteria = lambda k: (
            (layer_number is None or k[0] == layer_number)
            and (monitor_type is None or k[1] == monitor_type)
        )

        for key, monitor in self.monitors.items():
            if criteria(key):
                if enable:
                    if monitor not in self.network:
                        monitor.active = True
                else:
                    if monitor in self.network:
                        monitor.active = False
                toggled.append(f"{key[1]} on {key[0]}")

        if not toggled:
            return "No monitors matched the criteria."
        else:
            return (
                f"Monitors {'enabled' if enable else 'disabled'}: {', '.join(toggled)}"
            )

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

        if monitor_type == "spike":
            plot_spikes(monitor)
        elif monitor_type == "voltage":
            plot_voltages(monitor)
        elif monitor_type == "pop_avg_spike":
            plot_pop_avg_spikes(monitor)
        else:
            return "Unsupported monitor type."

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
        store = np.zeros((num_neurons, num_stimuli))
        edges = 0
        bins = np.arange(0, length_stimuli * (num_stimuli + 1), length_stimuli)
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

        # Check if data already exists
        if (layer, "spike", "histogram") not in self.monitor_data:
            store = self.bin_spikes(monitor, num_stimuli, length_stimuli)
            self.monitor_data[(layer, "spike", "histogram")] = store
        else:
            store = self.monitor_data[(layer, "spike", "histogram")]

        histogram = store[index, :]
        edges = np.arange(0, length_stimuli * (num_stimuli + 1), length_stimuli)
        plt.figure(figsize=(10, 6))  # Set the size of the figure
        plt.bar(
            edges[:-1], histogram, width=np.diff(edges), edgecolor="black", align="edge"
        )  # Create a bar plot

        plt.xlabel("Value Range")  # X-axis label
        plt.ylabel("Count")  # Y-axis label
        plt.title("Histogram of Values")  # Title of the plot
        plt.xticks(edges)  # Set x-ticks to match the edges
        plt.show()  # Display the plot

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

        def update(i):
            heatmap = self.generate_spike_heatmap(
                layer, type, num_stimuli, length_stimuli, layer_length, i
            )
            im.set_data(heatmap)
            ax.set_title(f"Spike Heatmap - Stimulus {i}")  # Update title
            return [im]

        # Set up the axes with column and row values
        ax.set_xticks(np.arange(layer_length))
        ax.set_yticks(np.arange(layer_length))
        ax.set_xticklabels(np.arange(1, layer_length + 1))
        ax.set_yticklabels(np.arange(1, layer_length + 1))

        ani = animation.FuncAnimation(fig, update, frames=range(num_stimuli), blit=True)
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
