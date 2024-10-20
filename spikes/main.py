# this is the main module. I will use this to gate running scripts as I develop. My aim is to develop clear well documented readable code.
from input import *
from tensorflow.keras.datasets import mnist

def fashion():
    # Exemplifies how to use the filter module 
    import numpy as np
    lambdas = [2]  # Wavelengths
    betas = [1.5]  # Scaling factor for bandwidth
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations
    psis = [0, np.pi]  # Phase offsets
    gammas = [0.5]  # Aspect ratio
    size = 128
    gabor_filters = GaborFilters(size, lambdas,betas,thetas,psis,gammas)

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Extract the first 30 images
    num_images = 30
    dataset = train_images[:num_images]

    # Normalize the dataset
    dataset = dataset.astype(np.float32) / 255.0
    neuron_inputs = generate_inputs_from_filters(dataset,
                                                 gabor_filters,
                                                 neuron_size = 14,
                                                 image_size = 28,
                                                 num_total_pixels=201,
                                                 radius=2,
                                                 shape="circle")







if __name__ == '__main__':
    # When run from the command line we gate what is then run using the sys module
    
    import sys
    # The first argument is the script name

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "p1":
            fashion()
    else:
        print("""select what programme to run:
              fashion: project1 - looks at the fashion mnist
              """)
