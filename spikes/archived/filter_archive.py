import numpy as np
from itertools import product
from PIL import Image
import os
import random 
from scipy.signal import convolve2d

class NeuronInputs:
    """
    Class holding the 3D array of input along with the mapping that was used to produce it
    """
    def __init__(self, input_train, mapping):
        self.input_train = input_train
        self.mapping = mapping

class GaborFilter:
    """
    Class representing a single Gabor filter. Responsible for generating, 
    manipulating, and saving the Gabor filter based on the input parameters.
    """
    
    def __init__(self, size, lambda_, beta, theta, psi, gamma):
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
        self.size = size
        self.lambda_ = lambda_
        self.beta = beta
        self.theta = theta
        self.psi = psi
        self.gamma = gamma
        self.filter_array = self._generate_filter()

    def _generate_meshgrid(self):
        """
        Generate the meshgrid (x, y coordinates) for the Gabor filter based on size.

        Returns:
        x, y (ndarray): Two 2D arrays representing the coordinates.
        """
        x = np.linspace(-self.size // 2, self.size // 2 - 1, self.size)
        y = np.linspace(-self.size // 2, self.size // 2 - 1, self.size)
        return np.meshgrid(x, y)

    def _generate_filter(self):
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
        sigma = (self.lambda_ * (2**self.beta + 1)) / (np.pi * (2**self.beta - 1))

        # Calculate the exponential (Gaussian) and cosine (sinusoidal) components
        exp_component = np.exp(-(x_prime**2 + self.gamma**2 * y_prime**2) / (2 * sigma**2))
        cos_component = np.cos(2 * np.pi * x_prime / self.lambda_ + self.psi)
        
        return exp_component * cos_component

    def save_as_image(self, directory):
        """
        Save the Gabor filter as a PNG image.

        Args:
        directory (str): Directory where the image will be saved.
        """
        filename = f"gabor_l{self.lambda_}_b{self.beta}_t{self.theta:.2f}_p{self.psi:.2f}_g{self.gamma}.png"
        filepath = os.path.join(directory, filename)

        # Normalize the filter values to the range [0, 255] for image saving
        normalized_filter = 255 * (self.filter_array - np.min(self.filter_array)) / (np.max(self.filter_array) - np.min(self.filter_array))
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
        for lambda_, beta, theta, psi, gamma in product(self.lambdas, self.betas, self.thetas, self.psis, self.gammas):
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
            if (gabor_filter.lambda_ == lambda_ and gabor_filter.beta == beta and 
                gabor_filter.theta == theta and gabor_filter.psi == psi and 
                gabor_filter.gamma == gamma):
                return gabor_filter
        # Raise an error if no matching filter is found
        raise ValueError(f"Gabor filter with parameters lambda={lambda_}, beta={beta}, theta={theta}, psi={psi}, gamma={gamma} not found.")

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
        self.conv_result = conv_result    # Store the convolution result

    def save_as_image(self, directory):
        """
        Save the convolution result as a PNG image.

        Args:
        directory (str): Directory where the image will be saved.
        """
        filename = f"conv_l{self.gabor_filter.lambda_}_b{self.gabor_filter.beta}_t{self.gabor_filter.theta:.2f}_p{self.gabor_filter.psi:.2f}_g{self.gabor_filter.gamma}.png"
        filepath = os.path.join(directory, filename)

        # Normalize the convolution result to [0, 255] for saving as an image
        normalized_result = 255 * (self.conv_result - np.min(self.conv_result)) / (np.max(self.conv_result) - np.min(self.conv_result))
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
        self.image = image          # Store the input image
        self.gabor_filters = gabor_filters  # Store the GaborFilters object
        self.layers = []            # List to store ConvLayer objects
        
        self._create_conv_layers()  # Automatically create ConvLayer objects

    def _create_conv_layers(self):
        """
        Create a ConvLayer for each Gabor filter by convolving the filter with the input image.
        """
        # Loop through all Gabor filters in the GaborFilters object
        for gabor_filter in self.gabor_filters.filters:
            # Perform the 2D convolution between the image and the Gabor filter
            conv_result = convolve2d(self.image, gabor_filter.filter_array, mode='same', boundary='wrap')

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
            if (gabor_filter.lambda_ == lambda_ and
                gabor_filter.beta == beta and
                gabor_filter.theta == theta and
                gabor_filter.psi == psi and
                gabor_filter.gamma == gamma):
                return layer
        
        # Raise an error if no matching ConvLayer is found
        raise ValueError(f"No ConvLayer found with Gabor filter parameters lambda={lambda_}, beta={beta}, theta={theta}, psi={psi}, gamma={gamma}")

def convolve_data_gabor(dataset, gabor_filters):
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
            convolved_image = convolve2d(image, gabor_filter.filter_array, mode='same', boundary='wrap')
            # Store the convolved image in the 4D array
            convolved_results[image_idx, :, :, filter_idx] = convolved_image

    return convolved_results

class ImageMapping:
    def __init__(self, mapping=None, image_size=None, neuron_size=None, num_layers=None, num_total_pixels=None, radius=None, shape=None):
        self.image_size = image_size
        self.neuron_size = neuron_size
        self.num_layers = num_layers
        self.num_total_pixels = num_total_pixels
        self.radius = radius
        self.shape = shape
        self.mapping = self.gen_mappings(image_size, neuron_size, num_layers, num_total_pixels, radius, shape)
        
    def gen_mappings(self, image_size, neuron_size, num_layers, num_total_pixels, radius, shape):
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
        scale = image_size // neuron_size  # Calculate scaling from neuron grid to image grid
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
                            distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                            # Only include pixels within the circular radius
                            if distance <= radius:
                                region_pixels.append((x, y))
                        elif shape == "square":
                            # Include all pixels within the bounding box for a square
                            region_pixels.append((x, y))
                
                # Get the total number of available pixels in the region
                available_pixels = len(region_pixels) * num_layers  # Number of 3D pixels (region size x number of layers)

                # Check if the requested number of pixels is greater than the available pixels
                if num_total_pixels > available_pixels:
                    print(f"Warning: Requested {num_total_pixels} pixels, but only {available_pixels} available for neuron ({neuron_x}, {neuron_y}).")
                    num_samples = available_pixels  # Limit the number of samples to the available pixels
                else:
                    num_samples = num_total_pixels

                # Set to store selected 3D coordinates (layer, x, y) without replacement
                selected_pixels = set()

                # Generate all possible combinations of (layer, x, y) coordinates
                all_3d_coords = [(layer, x, y) for layer in range(num_layers) for x, y in region_pixels]

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
            for (neuron_x, neuron_y) in image_mapping.keys():
                # Get the precomputed 3D pixel mappings for this neuron
                selected_pixels = image_mapping[(neuron_x, neuron_y)]

                # Collect pixel values from the convolved data based on the 3D coordinates
                input_values = []
                for layer, x, y in selected_pixels:
                    input_values.append(convolved_data[image_idx, x, y, layer])

                # Average the input values to assign to the neuron
                neuron_inputs[image_idx, neuron_x, neuron_y] = np.mean(input_values)

        return neuron_inputs

def generate_inputs_from_filters(dataset, gabor_filters, neuron_size, image_size, num_total_pixels=100, radius=6, shape="circle"):
    """
    Generate neuron inputs for a dataset by convolving images with Gabor filters and mapping pixels to neurons.

    Args:
    dataset (ndarray): 3D array of images with shape (num_images, height, width).
    gabor_filters (GaborFilters): GaborFilters object containing multiple filters.
    neuron_size (int): Size of the neuron grid (e.g., 14 for 14x14 neurons).
    image_size (int): Size of the input images (e.g., 28x28).
    num_total_pixels (int): Number of pixels to sample for each neuron.
    radius (int): Radius around the center of the pixel region for each neuron.
    shape (str): Shape of the region ("circle" or "square").

    Returns:
    3D ndarray: Neuron inputs with shape (num_images, neuron_size, neuron_size).
    """
    # Step 1: Convolve the dataset with the Gabor filters
    convolved_data = convolve_data_gabor(dataset, gabor_filters)

    # Step 2: Generate pixel mappings from the image space to the neuron grid
    num_layers = len(gabor_filters.filters)
    image_mapping = ImageMapping(image_size, neuron_size, num_layers, num_total_pixels, radius, shape)

    # Step 3: Generate neuron inputs using the 3D pixel mappings
    neuron_train = image_mapping.gen_inputs(convolved_data)
    neuron_inputs = NeuronInputs(neuron_train, image_mapping)
    return neuron_inputs
