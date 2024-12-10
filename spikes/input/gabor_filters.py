# gabor_filters.py

import numpy as np
from itertools import product
from PIL import Image
import os


class GaborFilter:
    """
    Class representing a single Gabor filter. Responsible for generating,
    manipulating, and saving the Gabor filter based on the input parameters.
    """

    def __init__(self, size, lambda_, beta, theta, psi, gamma, normalise=True):
        """
        Initialize a GaborFilter object with specific parameters.

        Args:
        size (int): Size of the square filter.
        lambda_ (float): Wavelength of the sinusoidal factor.
        beta (float): Scaling factor controlling bandwidth.
        theta (float): Orientation of the Gabor filter in radians.
        psi (float): Phase offset of the sinusoidal factor.
        gamma (float): Aspect ratio of the Gaussian envelope.
        """
        self.size = size  # Refactor this as not constant
        self.lambda_ = lambda_
        self.beta = beta
        self.theta = theta
        self.psi = psi
        self.gamma = gamma
        self.filter_array = self._generate_filter(normalise)

    def _generate_meshgrid(self):
        """
        Generate the meshgrid (x, y coordinates) for the Gabor filter based on size.

        Returns:
        x, y (ndarray): Two 2D arrays representing the coordinates.
        """
        x = np.linspace(-self.size // 2, self.size // 2 - 1, self.size)
        y = np.linspace(-self.size // 2, self.size // 2 - 1, self.size)
        return np.meshgrid(x, y)

    def _generate_filter(self, normalise=True):
        """
        Generate the Gabor filter based on the object's parameters.

        Returns:
        filter_array (ndarray): The 2D Gabor filter.
        """
        x, y = self._generate_meshgrid()
        # Apply rotation to coordinates
        x_prime = x * np.cos(self.theta) + y * np.sin(self.theta)
        y_prime = -x * np.sin(self.theta) + y * np.cos(self.theta)

        # Compute sigma (standard deviation) for the Gaussian component
        sigma = (
            (self.lambda_ * (2**self.beta + 1))
            / (np.pi * (2**self.beta - 1))
            * np.sqrt(np.log(2) / 2)
        )

        # Calculate the exponential (Gaussian) and cosine (sinusoidal) components
        exp_component = np.exp(
            -(x_prime**2 + self.gamma**2 * y_prime**2) / (2 * sigma**2)
        )
        cos_component = np.cos(2 * np.pi * x_prime / self.lambda_ + self.psi)

        filter = exp_component * cos_component
        # Does this work?
        if normalise:
            filter -= np.mean(filter)
            filter /= np.linalg.norm(filter)
        print(filter)
        return filter

    def save_as_image(self, directory):
        """
        Save the Gabor filter as a PNG image.

        Args:
        directory (str): Directory where the image will be saved.
        """
        filename = f"gabor_l{self.lambda_}_b{self.beta}_t{self.theta:.2f}_p{self.psi:.2f}_g{self.gamma}.png"
        filepath = os.path.join(directory, filename)

        # Normalize the filter values to the range [0, 255] for image saving
        normalized_filter = (
            255
            * (self.filter_array - np.min(self.filter_array))
            / (np.max(self.filter_array) - np.min(self.filter_array))
        )
        image = Image.fromarray(normalized_filter.astype(np.uint8))
        image.save(filepath)
        print(f"Saved image to {filepath}")

    def save_as_numpy(self, directory):
        """
        Save the Gabor filter as a NumPy array file (.npy).

        Args:
        directory (str): Directory where the NumPy file will be saved.
        """
        filename = f"gabor_l{self.lambda_}_b{self.beta}_t{self.theta:.2f}_p{self.psi:.2f}_g{self.gamma}.npy"
        filepath = os.path.join(directory, filename)
        np.save(filepath, self.filter_array)
        print(f"Saved NumPy array to {filepath}")


class GaborFilters:
    """
    Class responsible for generating, storing, and managing multiple Gabor filters
    based on various combinations of parameters.
    """

    def __init__(self, size, lambdas, betas, thetas, psis, gammas):
        """
        Initialize the GaborFilters object by generating and storing GaborFilter objects
        for each combination of input parameters.

        Args:
        size (int): Size of the square Gabor filters.
        lambdas (list of floats): List of wavelengths for the filters.
        betas (list of floats): List of scaling factors controlling bandwidth.
        thetas (list of floats): List of orientations (angles) in radians.
        psis (list of floats): List of phase offsets.
        gammas (list of floats): List of aspect ratios.
        """
        self.size = size
        self.lambdas = lambdas
        self.betas = betas
        self.thetas = thetas
        self.psis = psis
        self.gammas = gammas
        self.filters = []  # Store GaborFilter objects in a list
        self._generate_all_filters()

    def _generate_all_filters(self):
        """
        Generate all possible Gabor filters based on the provided parameter lists.
        Each unique combination of parameters is used to create a GaborFilter object.
        """
        # Use itertools.product to get every combination of the parameter values
        for lambda_, beta, theta, psi, gamma in product(
            self.lambdas, self.betas, self.thetas, self.psis, self.gammas
        ):
            # Create a new GaborFilter instance for each parameter combination
            gabor_filter = GaborFilter(self.size, lambda_, beta, theta, psi, gamma)
            self.filters.append(gabor_filter)

    def get_filter_by_params(self, lambda_, beta, theta, psi, gamma):
        """
        Retrieve a specific GaborFilter object based on its parameters.

        Args:
        lambda_ (float): Wavelength of the desired filter.
        beta (float): Scaling factor controlling bandwidth.
        theta (float): Orientation of the desired filter in radians.
        psi (float): Phase offset of the desired filter.
        gamma (float): Aspect ratio of the desired filter.

        Returns:
        GaborFilter: The Gabor filter object that matches the input parameters.

        Raises:
        ValueError: If no filter is found with the specified parameters.
        """
        # Search the filters list for a GaborFilter object with matching parameters
        for gabor_filter in self.filters:
            if (
                gabor_filter.lambda_ == lambda_
                and gabor_filter.beta == beta
                and gabor_filter.theta == theta
                and gabor_filter.psi == psi
                and gabor_filter.gamma == gamma
            ):
                return gabor_filter
        # Raise an error if no matching filter is found
        raise ValueError(
            f"Gabor filter with parameters lambda={lambda_}, beta={beta}, theta={theta}, psi={psi}, gamma={gamma} not found."
        )

    def store_all_as_images(self, directory):
        """
        Save all stored Gabor filters as PNG images in the specified directory.

        Args:
        directory (str): Directory where the images will be saved.
        """
        for gabor_filter in self.filters:
            gabor_filter.save_as_image(directory)

    def store_all_as_numpy(self, directory):
        """
        Save all stored Gabor filters as NumPy array files (.npy) in the specified directory.

        Args:
        directory (str): Directory where the NumPy files will be saved.
        """
        for gabor_filter in self.filters:
            gabor_filter.save_as_numpy(directory)
