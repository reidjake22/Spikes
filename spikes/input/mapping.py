# mapping.py
"""
Module Name: mapping.py
----------------------------------------------------

Purpose: 
--------
    This module provides functionality for generating neuron input mappings from images using Gabor filters.
    It includes classes and functions to create mappings, generate neuron inputs, and visualize the mappings.
    The mappings are used to convert image data into neuron input data for further processing.

Functions:
----------
    generate_inputs_from_filters:
        Generates neuron inputs for a dataset by convolving images with
    Gabor filters and mapping pixels to neurons.

    generate_timed_input_from_input:
        Generate a TimedArray from a 3D input array, where the input array
    is assumed to be in the format (num_images, height, width).

Classes:
--------
    NeuronInputs:
        Class holding the 3D array of input along with the mapping that was
    used to produce it.
    
    ImageMapping:
        Class to generate and manage pixel mappings from image space to neuron
    grid. Includes methods for generating mappings, generating neuron inputs, and visualizing mappings.
    
    OldImageMapping:
        Class to generate and manage pixel mappings from image space to neuron
    grid using an older method. Includes methods for generating mappings and generating neuron inputs.

Variables:
----------
    None

Example Usage:
--------------
    TODO - add examples


Notes:
--------------------
    TODO - add notes

"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from brian2 import TimedArray, hertz, ms

from .convolution import convolve_dataset_with_gabor_filters

from .gabor_filters import GaborFilters


class NeuronInputs:
    """
    Overview
    --------
        Class to store neuron inputs along with the pixel mappings used to generate them.

    Details:
    --------
        None

    Attributes:
    -----------
        input_train (ndarray):
            3D array of neuron inputs with shape (num_images, neuron_size, neuron_size).
        mapping (ImageMapping):
            ImageMapping object containing the pixel mappings used to generate the neuron inputs.
    Methods:
    --------
        visualise:
            Visualize the neuron inputs as a heatmap animation.


    Example Usage:
    --------
        None

    Notes:
    """

    def __init__(self, input_train, mapping):
        self.input_train = input_train
        self.mapping = mapping

    def visualise(self):
        # Sample data for demonstration (replace 'neuron_inputs' with your actual data array)
        fig, ax = plt.subplots()

        # Initialize a placeholder heatmap (use the first frame of data)
        heatmap = ax.imshow(self.input_train[0], cmap="hot", interpolation="nearest")
        ax.set_title("Layer 1")

        def update_heatmap(frame):
            ax.clear()
            # Update the data in the heatmap (not creating a new one)
            heatmap = ax.imshow(
                self.input_train[frame],
                cmap="hot",
                interpolation="nearest",
            )
            ax.set_title(f"Layer {frame + 1}")
            return (heatmap,)

        ani = animation.FuncAnimation(
            fig, update_heatmap, frames=len(self.input_train), blit=True
        )

        # Adding a colorbar corresponding to the heatmap
        plt.colorbar(heatmap, ax=ax)

        plt.show()


# Refactor later


class ImageMapping:
    """
    Overview
    --------
        Class to generate and manage pixel mappings from image space to neuron grid.
        Includes methods for generating mappings, generating neuron inputs, and visualizing mappings.
    Details:
    --------
        None

    Attributes:
    -----------
        image_size (int):
            The size of the image (assumed square).
        neuron_size (int):
            The size of the neuron array (assumed square).
        num_layers (int):
            The number of layers in the convolved image stack (number of filters).
        num_total_pixels (int):
            The total number of unique pixels to sample across layers (default: 100).
        radius (int):
            The radius around the center of the pixel region for each neuron (default: 6).
        shape (str):
            The shape of the eligible region, either "circle" or "square" (default: "circle").
        mapping (dict):
            A dictionary where each key is a (neuron_x, neuron_y) tuple and the value is
        a list of randomly selected (x, y, layer) coordinates.
    Methods:
    --------
        generate_single_neuron_mapping:
            generates a pixel mapping for a single neuron.

        gen_mappings:
            generates pixel mappings for all neurons in parallel.

        gen_inputs:
            generates neuron inputs from a 4D convolved dataset using precomputed 3D pixel mappings.

        visualize_neuron_mappings:
            visualizes the pixel mappings for a set of randomly selected neurons.
    Example Usage:
    --------------
        None
     Notes:
    -------
        None
    """

    def __init__(
        self,
        image_size=None,
        neuron_size=None,
        num_layers=None,
        num_total_pixels=None,
        radius=None,
        shape=None,
        mapping=None,
    ):
        """
        Initialize the ImageMapping object with the specified parameters & mapping or generate pixel mappings.

        arguments:
        ----------
            image_size (int): The size of the image (assumed square).
            neuron_size (int): The size of the neuron array (assumed square).
            num_layers (int): The number of layers in the convolved image stack (number of filters).
            num_total_pixels (int): The total number of unique pixels to sample across layers (default: 100).
            radius (int): The radius around the center of the pixel region for each neuron (default: 6).
            shape (str): The shape of the eligible region, either "circle" or "square" (default: "circle").
            mapping (dict): A dictionary where each key is a (neuron_x, neuron_y) tuple and the value is a list of randomly selected (x, y, layer) coordinates.

        returns:
        --------
            None

        Example Usage:
        --------------
            None

        Miscellaneous Notes:
        --------------------
            None
        """
        self.image_size = image_size
        self.neuron_size = neuron_size
        self.num_layers = num_layers
        self.num_total_pixels = num_total_pixels
        self.radius = radius
        self.shape = shape

        if mapping:
            self.mapping = mapping
        else:
            self.mapping = self.gen_mappings(
                image_size, neuron_size, num_layers, num_total_pixels, radius, shape
            )

    # Helper function to generate mappings for a single neuron
    def generate_single_neuron_mapping(
        self,
        neuron_x,
        neuron_y,
        image_size,
        neuron_size,
        num_layers,
        num_total_pixels,
        radius,
        shape,
    ):
        """
        Generate a pixel mapping for a single neuron (neuron_x, neuron_y) in the neuron grid.

        Arguments:
        ----------
        neuron_x (int): The x-coordinate of the neuron in the grid.
        neuron_y (int): The y-coordinate of the neuron in the grid.
        image_size (int): The size of the image (assumed square).
        neuron_size (int): The size of the neuron array (assumed square).
        num_layers (int): The number of layers in the convolved image stack (number of filters).
        num_total_pixels (int): The total number of unique pixels to sample across layers.
        radius (int): The radius around the center of the pixel region for each neuron.
        shape (str): The shape of the eligible region, either "circle" or "square".

        Returns:
        --------
        tuple: A tuple (neuron_x, neuron_y, selected_pixels) containing the neuron coordinates and the selected 3D pixels.

        Notes:
        ------
        None
        """
        scale = (
            image_size // neuron_size
        )  # Calculate scaling from neuron grid to image grid

        # Map the neuron (neuron_x, neuron_y) to the corresponding center in the image
        x_center = int(scale * neuron_x + scale / 2)
        y_center = int(scale * neuron_y + scale / 2)

        # Define the region bounds in the image, ensuring they stay within the image boundaries
        x_min = max(0, x_center - radius)
        x_max = min(image_size - 1, x_center + radius)
        y_min = max(0, y_center - radius)
        y_max = min(image_size - 1, y_center + radius)

        # Collect pixel coordinates based on the shape
        region_pixels = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if shape == "circle":
                    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                    if (
                        distance <= radius
                    ):  # Only include pixels within the circular radius
                        region_pixels.append((x, y))
                elif shape == "square":
                    region_pixels.append((x, y))

        # Get the total number of available pixels in the region
        available_pixels = (
            len(region_pixels) * num_layers
        )  # Number of 3D pixels (region size x number of layers)

        # Determine the number of pixels to sample
        if num_total_pixels > available_pixels:
            print(
                f"Warning: Requested {num_total_pixels} pixels, but only {available_pixels} available for neuron ({neuron_x}, {neuron_y})."
            )
            num_samples = available_pixels
        else:
            num_samples = num_total_pixels

        # Generate all possible combinations of (layer, x, y) coordinates
        all_3d_coords = [
            (layer, x, y) for layer in range(num_layers) for x, y in region_pixels
        ]

        # Randomly sample num_samples 3D coordinates
        selected_pixels = random.sample(all_3d_coords, num_samples)

        # Return the result as a tuple (neuron_x, neuron_y, selected_pixels)
        return (neuron_x, neuron_y, selected_pixels)

    # Modified method with parallelization and progress tracking
    def gen_mappings(
        self, image_size, neuron_size, num_layers, num_total_pixels, radius, shape
    ):
        """
        Generate a pixel mapping for neurons in parallel, where each neuron is mapped to a subset of 3D pixels.
        A progress bar is displayed to show the progress of mapping generation.

        Returns:
        dict: A dictionary where each key is a (neuron_x, neuron_y) tuple and the value is
              a list of randomly selected (x, y, layer) coordinates.
        """
        pixel_mappings = {}

        # Use ProcessPoolExecutor to parallelize the task of generating mappings for each neuron
        with ProcessPoolExecutor() as executor:
            futures = []
            for neuron_x in range(neuron_size):
                for neuron_y in range(neuron_size):
                    futures.append(
                        executor.submit(
                            self.generate_single_neuron_mapping,
                            neuron_x,
                            neuron_y,
                            image_size,
                            neuron_size,
                            num_layers,
                            num_total_pixels,
                            radius,
                            shape,
                        )
                    )

            # Track progress with tqdm
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating neuron mappings",
            ):
                neuron_x, neuron_y, selected_pixels = future.result()
                pixel_mappings[(neuron_x, neuron_y)] = selected_pixels

        return pixel_mappings

    def gen_inputs(self, convolved_data):
        """
        Generate neuron inputs from a 4D convolved dataset using precomputed 3D pixel mappings.

        Args:
        convolved_data (ndarray): 4D array of convolved images with shape (num_images, height, width, num_filters).
        pixel_mappings (obj): ImageMapping object - contains the correct
        neuron_size (int): The size of the neuron array (e.g., 14 for a 14x14 neuron grid).

        Returns:
        3D ndarray: Neuron inputs with shape (num_images, neuron_size, neuron_size) Here it's image_idx, neuron_y, neuron_x.

        Notes:
        Basically it flips between matrix and cartesian coordinates, this is a bit confusing but hopefully it works out. The main thing is making sure that the right inputs are going to the right neurons
        """
        num_images, image_height, image_width, num_filters = convolved_data.shape
        neuron_size = self.neuron_size
        # Initialize the 3D array to store neuron inputs (num_images, neuron_size, neuron_size)
        neuron_inputs = np.zeros((num_images, neuron_size, neuron_size))

        # Loop over each image
        for image_idx in range(num_images):
            # Loop over each neuron
            for neuron_x, neuron_y in self.mapping.keys():
                # Get the precomputed 3D pixel mappings for this neuron
                selected_pixels = self.mapping[(neuron_x, neuron_y)]

                # Collect pixel values from the convolved data based on the 3D coordinates
                input_values = np.array([])

                # Loop over selected_pixels and append values from convolved_data
                for layer, x, y in selected_pixels:
                    input_values = np.append(
                        # convolved_data is in the format (num_images, height, width, num_filters)
                        input_values,
                        convolved_data[image_idx, y, x, layer],
                    )

                # Add the input values to assign to the neuron, sum only positive values
                neuron_inputs[image_idx, neuron_y, neuron_x] = (
                    np.sum(input_values[input_values > 0]) * 100
                )  # Scale factor 100
        return neuron_inputs

    def visualize_neuron_mappings(self, num_neurons=10):
        """
        Visualize the pixel mappings for a set of randomly selected neurons.
        For each neuron, the selected pixels are shown, the center pixel is highlighted in blue,
        and a blue boundary around the selection region is drawn.

        Args:
        num_neurons (int): Number of random neurons to visualize (default: 10).
        """
        import matplotlib.pyplot as plt
        import random
        import matplotlib.patches as patches

        if len(self.mapping) == 0:
            print("No mappings available to visualize.")
            return

        # Randomly select neuron coordinates to visualize
        random_neurons = random.sample(list(self.mapping.keys()), num_neurons)

        # Iterate over the randomly selected neurons
        for neuron_x, neuron_y in random_neurons:
            selected_pixels = self.mapping[(neuron_x, neuron_y)]

            # Initialize the base heatmap (empty image)
            heatmap = np.zeros((self.image_size, self.image_size))

            # Mark the selected pixels in the heatmap
            for layer, x, y in selected_pixels:
                heatmap[y, x] += 1  # Correct x and y alignment for heatmap

            # Visualize the heatmap
            plt.figure(figsize=(6, 6))
            plt.title(f"Neuron ({neuron_x}, {neuron_y}) - Pixel Mapping")

            # Show the heatmap in grey and red
            plt.imshow(heatmap, cmap="Reds", interpolation="nearest")

            # Invert y-axis to correct for imshow's top-left origin behavior
            plt.gca().invert_yaxis()

            # Highlight the neuron center pixel in blue
            scale = self.image_size // self.neuron_size
            x_center = int(scale * neuron_x + scale / 2)
            y_center = int(scale * neuron_y + scale / 2)
            plt.scatter(
                x_center, y_center, color="blue", label="Neuron Center", s=100
            )  # No need to swap x and y here

            # Define the region bounds for the neuron selection
            x_min = max(0, x_center - self.radius)
            x_max = min(self.image_size - 1, x_center + self.radius)
            y_min = max(0, y_center - self.radius)
            y_max = min(self.image_size - 1, y_center + self.radius)

            # Draw a blue boundary around the region (correct x and y alignment)
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.5,
                edgecolor="blue",
                facecolor="none",
                label="Selection Boundary",
            )
            plt.gca().add_patch(rect)

            # Add legend and show the plot
            plt.legend(loc="upper right")
            plt.xlabel("x-coordinate")
            plt.ylabel("y-coordinate")
            plt.show()


class OldImageMapping:
    def __init__(
        self,
        mapping=None,
        image_size=None,
        neuron_size=None,
        num_layers=None,
        num_total_pixels=None,
        radius=None,
        shape=None,
    ):
        self.image_size = image_size
        self.neuron_size = neuron_size
        self.num_layers = num_layers
        self.num_total_pixels = num_total_pixels
        self.radius = radius
        self.shape = shape
        self.mapping = self.gen_mappings(
            image_size, neuron_size, num_layers, num_total_pixels, radius, shape
        )

    def gen_mappings(
        self, image_size, neuron_size, num_layers, num_total_pixels, radius, shape
    ):
        """
        Generate a pixel mapping for neurons where each neuron is mapped to a subset of 3D pixels
        (x, y, layer) from an image, within a radius around its corresponding position. The region can be
        either circular or square.

        Args:
        image_size (int): The size of the image (assumed square).
        neuron_size (int): The size of the neuron array (assumed square).
        num_layers (int): The number of layers in the convolved image stack (number of filters).
        num_total_pixels (int): The total number of unique pixels to sample across layers (default: 100).
        radius (int): The radius around the neuron center in the image (default: 6).
        shape (str): The shape of the eligible region, either "circle" or "square" (default: "circle").

        Returns:
        dict: A dictionary where each key is a (neuron_x, neuron_y) tuple and the value is
                a list of randomly selected (x, y, layer) coordinates.
        """
        scale = (
            image_size // neuron_size
        )  # Calculate scaling from neuron grid to image grid
        pixel_mappings = {}

        # Iterate over each neuron in the neuron grid
        for neuron_x in range(neuron_size):
            for neuron_y in range(neuron_size):
                # Map the neuron (neuron_x, neuron_y) to the corresponding center in the image
                x_center = int(scale * neuron_x + scale / 2)
                y_center = int(scale * neuron_y + scale / 2)

                # Define the region bounds in the image, ensuring they stay within the image boundaries
                x_min = max(0, x_center - radius)
                x_max = min(image_size - 1, x_center + radius)
                y_min = max(0, y_center - radius)
                y_max = min(image_size - 1, y_center + radius)

                # Collect pixel coordinates based on the shape
                region_pixels = []
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        if shape == "circle":
                            # Compute the Euclidean distance from the center
                            distance = np.sqrt(
                                (x - x_center) ** 2 + (y - y_center) ** 2
                            )
                            # Only include pixels within the circular radius
                            if distance <= radius:
                                region_pixels.append((x, y))
                        elif shape == "square":
                            # Include all pixels within the bounding box for a square
                            region_pixels.append((x, y))

                # Get the total number of available pixels in the region
                available_pixels = (
                    len(region_pixels) * num_layers
                )  # Number of 3D pixels (region size x number of layers)

                # Check if the requested number of pixels is greater than the available pixels
                if num_total_pixels > available_pixels:
                    print(
                        f"Warning: Requested {num_total_pixels} pixels, but only {available_pixels} available for neuron ({neuron_x}, {neuron_y})."
                    )
                    num_samples = available_pixels  # Limit the number of samples to the available pixels
                else:
                    num_samples = num_total_pixels

                # Set to store selected 3D coordinates (layer, x, y) without replacement
                selected_pixels = set()

                # Generate all possible combinations of (layer, x, y) coordinates
                all_3d_coords = [
                    (layer, x, y)
                    for layer in range(num_layers)
                    for x, y in region_pixels
                ]

                # Randomly sample num_samples 3D coordinates without replacement
                selected_pixels = random.sample(all_3d_coords, num_samples)

                # Store the selected 3D coordinates for this neuron
                pixel_mappings[(neuron_x, neuron_y)] = list(selected_pixels)

        return pixel_mappings

    def gen_inputs(self, convolved_data):
        """
        Generate neuron inputs from a 4D convolved dataset using precomputed 3D pixel mappings.

        Args:
        convolved_data (ndarray): 4D array of convolved images with shape (num_images, height, width, num_filters).
        pixel_mappings (obj): ImageMapping object - contains the correct
        neuron_size (int): The size of the neuron array (e.g., 14 for a 14x14 neuron grid).

        Returns:
        3D ndarray: Neuron inputs with shape (num_images, neuron_size, neuron_size).
        """
        num_images, image_height, image_width, num_filters = convolved_data.shape
        neuron_size = self.neuron_size
        image_mapping = self.mapping
        # Initialize the 3D array to store neuron inputs (num_images, neuron_size, neuron_size)
        neuron_inputs = np.zeros((num_images, neuron_size, neuron_size))

        # Loop over each image
        for image_idx in range(num_images):
            # Loop over each neuron
            for neuron_x, neuron_y in image_mapping.keys():
                # Get the precomputed 3D pixel mappings for this neuron
                selected_pixels = image_mapping[(neuron_x, neuron_y)]

                # Collect pixel values from the convolved data based on the 3D coordinates
                input_values = []
                for layer, x, y in selected_pixels:
                    input_values.append(convolved_data[image_idx, x, y, layer])

                # Average the input values to assign to the neuron
                neuron_inputs[image_idx, neuron_x, neuron_y] = np.mean(input_values)

        return neuron_inputs


def generate_inputs_from_filters(
    dataset,
    gabor_filters,
    neuron_size,
    image_size,
    num_total_pixels=100,
    radius=6,
    shape="circle",
):
    """
    Generate neuron inputs for a dataset by convolving images with Gabor filters and mapping pixels to neurons.

    Args:
    dataset (ndarray): 3D array of images with shape (num_images, height, width).
    Basically it's channel first
    gabor_filters (GaborFilters): GaborFilters object containing multiple filters.
    neuron_size (int): Size of the neuron grid (e.g., 14 for 14x14 neurons).
    image_size (int): Size of the input images (e.g., 28x28).
    num_total_pixels (int): Number of pixels to sample for each neuron.
    radius (int): Radius around the center of the pixel region for each neuron.
    shape (str): Shape of the region ("circle" or "square").

    Returns:
    NeuronInputs: NeuronInputs object containing the 3D array of neuron inputs and the pixel mappings.
    """
    # Step 1: Convolve the dataset with the Gabor filters
    convolved_data = convolve_dataset_with_gabor_filters(dataset, gabor_filters)

    # Step 2: Generate pixel mappings from the image space to the neuron grid
    num_layers = len(gabor_filters.filters)
    image_mapping = ImageMapping(
        image_size, neuron_size, num_layers, num_total_pixels, radius, shape
    )

    # Step 3: Generate neuron inputs using the 3D pixel mappings
    neuron_train = image_mapping.gen_inputs(convolved_data)
    neuron_inputs = NeuronInputs(neuron_train, image_mapping)
    return neuron_inputs


def generate_timed_input_from_input(
    neuron_inputs: NeuronInputs, stimulus_exposure_time: int
):
    """
    Generate a TimedArray from a 3D input array, where the input array is assumed to be
    in the format (num_images, height, width).

    Args:
    input (ndarray): 3D input array with shape (num_images, height, width).

    Returns:
    TimedArray: TimedArray object with the input values.
    """
    neuron_input_train = neuron_inputs.input_train
    index, height, width = neuron_input_train.shape
    print(
        f"the number of images for the input is {index}, the height is {height}, and the width is {width}"
    )

    # Gonna have to sort the dimensions here unfortunately - what a hassle
    collapsed_input = np.array(
        [neuron_input_train.flatten() for image in neuron_input_train]
    )

    collapsed_input_hz = collapsed_input * hertz
    timed_input = TimedArray(collapsed_input_hz, dt=stimulus_exposure_time)
    return timed_input


def generate_3d_poisson_rates_from_filters(
    dataset,
    gabor_filters,
    neuron_size,
    image_size,
    num_total_pixels=100,
    radius=6,
    normalise=True,
):
    """
    Generates a 4D array that corresponds to a 3D poisson train for each image in the dataset.

    Args:
    dataset (ndarray): 3D array of images with shape (num_images, height, width).
    gabor_filters (GaborFilters): GaborFilters object containing multiple filters.
    neuron_size (int): Size of the neuron grid (e.g., 14 for 14x14 neurons).
    image_size (int): Size of the input images (e.g., 28x28).
    num_total_pixels (int): Number of pixels to sample for each neuron.
    radius (int): Radius around the center of the pixel region for each neuron.
    Returns:
    ndarray: 4D array of poisson rates with shape (num_images, neuron_size, neuron_size, num_filters).
    """

    # Step 1: Convolve the dataset with the Gabor filters
    convolved_data = convolve_dataset_with_gabor_filters(dataset, gabor_filters)

    # Step 2: Normalise each convolved image using the euclidian norm
    if normalise:
        # Compute the norm along spatial dimensions (1, 2)
        norm = np.linalg.norm(
            convolved_data, axis=(1, 2), keepdims=True
        )  # Shape: (30, 1, 1, 8)

        # Normalize by broadcasting the norm across spatial dimensions
        convolved_data = convolved_data / norm

    return convolved_data


def generate_flat_poisson_inputs_from_convolved_data(convolved_data):
    """
    Create a 2D array of inputs where each row corresponds to a flattened 3D poisson train for each image in the dataset. and each column corresponds to an image

    it basically gives value for each coordinate for each filter, then moves along the row then moves along the column
    """

    num_images, image_height, image_width, num_filters = convolved_data.shape
    new_array = np.zeros((num_images, image_height * image_width * num_filters))
    for image_idx in range(num_images):
        new_array[image_idx] = convolved_data[image_idx].flatten()
    return new_array


def generate_timed_array_from_flat_poisson_inputs(
    poisson_inputs,
    beta,
    stimulus_exposure_time,
):
    """
    Generate a TimedArray from a 2D input array, where the input array is assumed to be
    in the format (num_images, height*width*num_filters).

    Args:
    input (ndarray): 2D input array with shape (num_images, height*width*num_filters).

    Returns:
    TimedArray: TimedArray object with the input values.
    """
    num_images, num_neurons = poisson_inputs.shape
    collapsed_input_hz = poisson_inputs * beta * hertz
    print(f"beta:{beta}")
    print(collapsed_input_hz)
    timed_input = TimedArray(collapsed_input_hz, dt=stimulus_exposure_time)
    return timed_input
