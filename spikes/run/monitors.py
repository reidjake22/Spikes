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
    def __init__(self, network, n_layers):
        self.network = network
        self.n_layers = n_layers
        self.monitors = {}  # Dictionary to store monitors by (layer_name, monitor_type)
        self.monitor_data = (
            {}
        )  # Dictionary to store monitor data by (layer_name, monitor_type, data_type)

    def create_monitor(self, neuron_group, monitor_type, layer, **kwargs):
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
        for layer in layers:
            layer_name = (
                f"excitatory_layer_{layer}" if layer != 0 else "poisson_layer_0"
            )
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

    def toggle_monitoring(self, layer_number=None, monitor_type=None, enable=True):
        """
        Toggle monitoring for specified layer and/or monitor type across all matches.
        If both layer_name and monitor_type are None, toggles all monitors.
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
        Retrieve monitors based on layer name and/or monitor type.
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
        Visualise monitor data.
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
        spikes = monitor.spike_trains()
        num_neurons = monitor.source.N
        store = np.zeros((num_neurons, num_stimuli))
        edges = 0
        bins = np.arange(0, length_stimuli * (num_stimuli + 1), length_stimuli)
        for key, value in spikes.items():
            print("binning")
            counts, edges = np.histogram(value, bins=bins)
            store[key] = counts
        # Ideally would happen here but layer is inaccessible
        return store

    def plot_spikes(self, layer, type, index, num_stimuli, length_stimuli):
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
        heatmap = self.generate_spike_heatmap(
            layer, type, num_stimuli, length_stimuli, layer_length, stimulus_index
        )
        plt.imshow(heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.show()

    def animate_spike_heatmap(
        self, layer, type, num_stimuli, length_stimuli, layer_length
    ):
        fig, ax = plt.subplots()
        heatmap = self.generate_spike_heatmap(
            layer, type, num_stimuli, length_stimuli, layer_length, 0
        )
        im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax)  # Add color bar
        ax.set_title(f"Spike Heatmap - Stimulus 0")  # Initial title

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
