import numpy as np

def perform_traceback():
    no_images = 20,000
    no_neurons = 128 * 128
    activity_of_final_layer_neurons = np.zeros((20000, 128, 128))
    summary_images = np.zeros((128*128, 128, 128))
    inputs_to_layer_1 = np.zeros((20000, 128, 128))
    for neuron in range(no_neurons):
        

