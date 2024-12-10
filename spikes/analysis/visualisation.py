from brian2 import *
from network import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from .analysis import determine_inputs_per_image


def plot_synapse_distribution(synapses: SynapseSpecs, name: str):
    """
    Plots the distribution of synapse weights for a specific synapse object.

    Parameters:
    -----------
    synapse_name : str
        The name of the synapse object to plot the distribution for.
    """
    synapses = synapses.synapse_objects[name]
    plt.figure(figsize=(6, 6))
    plt.hist(synapses.w, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Synapse Weight Distribution for {name}")
    plt.xlabel("Synapse Weight")
    plt.ylabel("Frequency")
    plt.show()


def plot_synapse_distributions(synapses: SynapseSpecs):
    """
    Plots the distribution of synapse weights for all synapse objects.

    Parameters:
    -----------
    synapses : SynapseSpecs
        The synapse object containing all synapse information.
    """
    # Get connected layers and their corresponding synapse objects
    synapse_objects = synapses.synapse_objects
    connected_layers = list(synapse_objects.keys())
    n_layers = len(connected_layers)

    if n_layers == 0:
        raise ValueError("No synapse objects found in synapse_specs.")

    # Create subplots
    fig, axs = plt.subplots(
        n_layers,
        1,
        figsize=(6, min(6 * n_layers, 24)),  # Cap total height at 24
        constrained_layout=True,
    )

    # Handle the case of a single subplot (not returned as a list)
    if n_layers == 1:
        axs = [axs]

    # Iterate over each layer and plot its distribution
    for i, (layer_name, synapse) in enumerate(synapse_objects.items()):
        if not hasattr(synapse, "w") or len(synapse.w) == 0:
            print(f"Skipping layer '{layer_name}' (no weights to plot).")
            continue

        ax = axs[i]
        ax.hist(synapse.w, bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Synapse Weight Distribution for Layer '{layer_name}'")
        ax.set_xlabel("Synapse Weight")
        ax.set_ylabel("Frequency")

    # Show the plot
    plt.show()


def visualise_synapses(synapses: Synapses):
    Ns = len(synapses.source)
    Nt = len(synapses.target)
    figure(figsize=(20, 10))
    subplot(121)
    plot(zeros(Ns), arange(Ns), "ok", ms=10)
    plot(ones(Nt), arange(Nt), "ok", ms=10)
    for i, j in zip(synapses.i, synapses.j):
        plot([0, 1], [i, j], "-k")
    xticks([0, 1], ["Source", "Target"])
    ylabel("Neuron index")
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(synapses.i, synapses.j, "ok")
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel("Source neuron index")
    ylabel("Target neuron index")


def three_dim_visualise_synapses_illustrate(synapses: Synapses):
    Ns = len(synapses.source)
    num_pre_neurons = len(synapses.N_outgoing_pre)
    len_pre = int(sqrt(num_pre_neurons))
    Nt = len(synapses.target)
    num_post_neurons = len(synapses.N_incoming_post)
    len_post = int(sqrt(num_post_neurons))
    s_i_column = synapses.i % len_pre
    s_i_row = synapses.i // len_pre
    s_j_column = synapses.j % len_post
    s_j_row = synapses.j // len_post
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        s_i_column, s_i_row, 0, c="blue", label="Pre-synaptic neurons", s=100
    )  # Blue circles
    ax.scatter(
        s_j_column, s_j_row, 1, c="red", label="Post-synaptic neurons", s=100
    )  # Red circles
    for x1, y1, x2, y2 in zip(s_i_column, s_i_row, s_j_column, s_j_row):
        if (x1, y1) == (len_pre // 4, len_pre // 4):
            ax.plot(
                [x1, x2], [y1, y2], [0, 1], "-k", alpha=0.9, color="gold"
            )  # gold lines with transparency
        elif (x2, y2) == (3 * len_pre // 4, 3 * len_pre // 4):
            ax.plot([x1, x2], [y1, y2], [0, 1], "-k", alpha=0.9, color="green")
        else:
            ax.plot([x1, x2], [y1, y2], [0, 1], "-k", alpha=0.1)
    plt.show()


def three_dim_visualise_synapses(synapses: Synapses):
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


def visualise_poisson_inputs(_3d_poisson_rates):
    num_images, neuron_size, _, num_filters = _3d_poisson_rates.shape

    # Aggregate across the filter dimension by summing all positive values
    collapsed_data = np.sum(
        np.where(_3d_poisson_rates > 0, _3d_poisson_rates, 0), axis=-1
    )

    # Set up the figure
    fig, ax = plt.subplots()
    ims = []

    vmin = collapsed_data.min()
    vmax = collapsed_data.max()

    # Create the animation frames
    for i in range(collapsed_data.shape[0]):
        im = ax.imshow(
            collapsed_data[i], animated=True, vmin=vmin, vmax=vmax, cmap="hot"
        )
        ims.append([im])

    # Create the animation
    ani = animation.ArtistAnimation(
        fig, ims, interval=500, blit=True, repeat_delay=100000
    )
    plt.colorbar(im, ax=ax)
    plt.show()


def display_input_activity(DATA, flat_poisson_inputs):
    image_activity = determine_inputs_per_image(DATA, flat_poisson_inputs)
    num_images = 8
    for i in range(num_images):
        print(f"image no. {i}")
        plt.figure(figsize=(5, 5))
        plt.title("Input Activity for Each Image")
        poisson_inputs_by_image = image_activity[i, :]
        grid = np.reshape(poisson_inputs_by_image, (64, 64)) / hertz

        print(f"grid shape is:{grid.shape}")
        print(grid.shape)
        plt.imshow(grid, cmap="hot", interpolation="nearest")
        plt.title(f"Image {i}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Neuron Index")
        plt.show()
