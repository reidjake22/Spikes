# convolution.py

import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from .gabor_filters import GaborFilter, GaborFilters  # Import from gabor_filters.py


class ConvLayer:
    """
    Class representing a single convolution layer.
    It stores the result of convolving an image with a Gabor filter,
    and keeps track of the GaborFilter used for the convolution.
    """

    def __init__(self, gabor_filter, conv_result):
        """
        Initialize the ConvLayer with a GaborFilter object and the convolution result.

        Args:
        gabor_filter (GaborFilter): The GaborFilter used for the convolution.
        conv_result (ndarray): The result of convolving the image with the Gabor filter.
        """
        self.gabor_filter = gabor_filter  # Store the GaborFilter used
        self.conv_result = conv_result  # Store the convolution result

    def save_as_image(self, directory):
        """
        Save the convolution result as a PNG image.

        Args:
        directory (str): Directory where the image will be saved.
        """
        filename = f"conv_l{self.gabor_filter.lambda_}_b{self.gabor_filter.beta}_t{self.gabor_filter.theta:.2f}_p{self.gabor_filter.psi:.2f}_g{self.gabor_filter.gamma}.png"
        filepath = os.path.join(directory, filename)

        # Normalize the convolution result to [0, 255] for saving as an image
        normalized_result = (
            255
            * (self.conv_result - np.min(self.conv_result))
            / (np.max(self.conv_result) - np.min(self.conv_result))
        )
        image = Image.fromarray(normalized_result.astype(np.uint8))
        image.save(filepath)
        print(f"Saved convolution result to {filepath}")


class ConvImage:
    """
    Class representing a collection of convolution layers.
    It stores multiple ConvLayer objects, each representing the result
    of convolving an image with a different Gabor filter.
    On initialization, all ConvLayer objects are created using the input image and GaborFilters.
    """

    def __init__(self, gabor_filters, image):
        """
        Initialize the ConvImage by performing convolutions of the input image with each GaborFilter.

        Args:
        gabor_filters (GaborFilters): The GaborFilters object containing multiple Gabor filters.
        image (ndarray): The input image to be convolved with each Gabor filter.
        """
        self.image = image  # Store the input image
        self.gabor_filters = gabor_filters  # Store the GaborFilters object
        self.layers = []  # List to store ConvLayer objects

        self._create_conv_layers()  # Automatically create ConvLayer objects

    def _create_conv_layers(self):
        """
        Create a ConvLayer for each Gabor filter by convolving the filter with the input image.
        """
        # Loop through all Gabor filters in the GaborFilters object
        for gabor_filter in self.gabor_filters.filters:
            # Perform the 2D convolution between the image and the Gabor filter
            conv_result = convolve2d(
                self.image, gabor_filter.filter_array, mode="same", boundary="wrap"
            )

            # Create a ConvLayer with the Gabor filter and convolution result
            conv_layer = ConvLayer(gabor_filter, conv_result)

            # Add the ConvLayer to the list of layers
            self.layers.append(conv_layer)

    def store_all_as_images(self, directory):
        """
        Save all convolution results (layers) as images in the specified directory.

        Args:
        directory (str): Directory where the images will be saved.
        """
        for layer in self.layers:
            layer.save_as_image(directory)

    def get_layer_by_filter_params(self, lambda_, beta, theta, psi, gamma):
        """
        Retrieve a ConvLayer by the parameters of the Gabor filter used.

        Args:
        lambda_ (float): Wavelength of the desired Gabor filter.
        beta (float): Scaling factor controlling bandwidth.
        theta (float): Orientation of the desired filter in radians.
        psi (float): Phase offset of the desired filter.
        gamma (float): Aspect ratio of the desired filter.

        Returns:
        ConvLayer: The ConvLayer object corresponding to the Gabor filter with the specified parameters.

        Raises:
        ValueError: If no matching ConvLayer is found.
        """
        # Iterate through the list of ConvLayer objects
        for layer in self.layers:
            gabor_filter = layer.gabor_filter
            # Check if the Gabor filter parameters match the desired parameters
            if (
                gabor_filter.lambda_ == lambda_
                and gabor_filter.beta == beta
                and gabor_filter.theta == theta
                and gabor_filter.psi == psi
                and gabor_filter.gamma == gamma
            ):
                return layer

        # Raise an error if no matching ConvLayer is found
        raise ValueError(
            f"No ConvLayer found with Gabor filter parameters lambda={lambda_}, beta={beta}, theta={theta}, psi={psi}, gamma={gamma}"
        )


# Helper function to convolve a single image with all Gabor filters
def convolve_image_with_gabor_filters(image, gabor_filters):
    num_filters = len(gabor_filters.filters)
    image_height, image_width = image.shape
    convolved_image = np.zeros((image_height, image_width, num_filters))

    # Convolve the image with each Gabor filter
    for filter_idx, gabor_filter in enumerate(gabor_filters.filters):
        convolved_image[:, :, filter_idx] = convolve2d(
            image, gabor_filter.filter_array, mode="same", boundary="wrap"
        )

    return convolved_image


# Parallelized function with progress tracking
def convolve_dataset_with_gabor_filters(dataset, gabor_filters):
    """
    Parallelize the convolution of each image in a dataset with a set of Gabor filters.
    A progress bar is displayed using tqdm, and parallel execution is handled by ProcessPoolExecutor.

    Args:
    dataset (ndarray): 3D array of images with shape (num_images, height, width).
    gabor_filters (GaborFilters): GaborFilters object containing multiple filters.

    Returns:
    4D ndarray: Convolved results with shape (num_images, height, width, num_filters).
    """
    num_images, image_height, image_width = dataset.shape
    # Add a little debugger or
    num_filters = len(gabor_filters.filters)

    # Initialize the 4D array to store convolved images
    convolved_results = np.zeros((num_images, image_height, image_width, num_filters))

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor() as executor:
        # Submit each image to be convolved in parallel
        futures = {
            executor.submit(
                convolve_image_with_gabor_filters, image, gabor_filters
            ): idx
            for idx, image in enumerate(dataset)
        }

        # Use tqdm to display a progress bar and process results as they are completed
        for future in tqdm(
            as_completed(futures), total=num_images, desc="Convolving images"
        ):
            image_idx = futures[future]
            convolved_results[image_idx] = future.result()

    return convolved_results


def old_convolve_dataset_with_gabor_filters(dataset, gabor_filters):
    """
    Convolve each image in a dataset with a set of Gabor filters and return a 4D array.
    This is the crux of it - what we do with this convolved set is combined it with the mapping
    Args:
    dataset (ndarray): 3D array of images with shape (num_images, height, width).
    gabor_filters (GaborFilters): GaborFilters object containing multiple filters.

    Returns:
    4D ndarray: Convolved results with shape (num_images, height, width, num_filters).
    """
    num_images, image_height, image_width = dataset.shape
    num_filters = len(gabor_filters.filters)

    # Initialize the 4D array to store convolved images
    convolved_results = np.zeros((num_images, image_height, image_width, num_filters))

    # Loop over each image in the dataset
    for image_idx, image in enumerate(dataset):
        # Loop over each Gabor filter
        for filter_idx, gabor_filter in enumerate(gabor_filters.filters):
            # Convolve the image with the current Gabor filter
            convolved_image = convolve2d(
                image, gabor_filter.filter_array, mode="same", boundary="wrap"
            )
            # Store the convolved image in the 4D array
            convolved_results[image_idx, :, :, filter_idx] = convolved_image

    return convolved_results
