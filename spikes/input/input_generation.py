from .gabor_filters import GaborFilters
from .convolution import convolve_dataset_with_gabor_filters
import os
import numpy as np
def generate_inputs(
        project_name: str,
        project_base: str,
        data_dir: str=None,
        gabor_filters: GaborFilters=None,
        filtered_input: np.ndarray=None,
        connectivity_settings: dict[int, int]=None, # radius, output_image_size

):
    filtered_home = False
    if filtered_input is None:
        try:
            filtered_input = np.load(os.path.join(project_base, "config/input/filtered_input.npy"))
            filtered_home = True
        except FileNotFoundError:
            filtered_input = None

    if GaborFilters is None and filtered_input is None:
        print("no gabor filters or filtered input provided, exiting")
        return
    elif GaborFilters:
        print("Gabor filters present")
        if data_dir is None:
            data_dir = os.path.join(project_base, "data")
        data = None
        try:
            data = np.load(os.path.join(data_dir, "input_data.npy"))
        except:
            raise FileNotFoundError("data.npy not found in data directory")
        if filtered_input and not filtered_home:
            print("why pass custom filtered input and gabor filters? Dumb")
            return
        if filtered_home:
            print("filtered input found from {base_dir}/config/input/filtered_input.npy")
            print("you will overwrite it if you continue")
            #let user decide
            cont = input("continue? (y/n)")
            if cont == "n":
                return
        filtered_input = convolve_dataset_with_gabor_filters(data, gabor_filters) # placeholder for filtered input 4D
        np.save(os.path.join(project_base, "config/input/filtered_input.npy"), filtered_input)
        print("filtered input saved to {base_dir}/config/input/filtered_input.npy")

    else:
        if filtered_home:
            print( "Gabor filters not present but filtered input found from {base_dir}/config/input/filtered_input.npy")
        else:
            print("custom input provided, all output will still be saved to {base_dir}/config/input/ ")
    # This is where we take filtered_input and convolve it according to connectivity
    try:
        mapping = np.load(os.path.join(project_base, "config/input/mapping.npy"))
        print(" there is a mapping found - if you continue having passed config settings it will be overwritten")
        cont = input("continue? (y/n)")
        if cont == "n":
            return
    except:
        pass
    if connectivity_settings is None:
        print("connectivity settings are none - try running again whilst passing them. Omiting Gabor filters and filtered input will mean you use the same data")
        return
    else:
        image_length = filtered_input.shape[1]
        connectivity = produce_connectivity(image_length, connectivity_settings)
        # save connectivity settings
    mapped_data = map_data(filtered_input, connectivity)

def produce_connectivity(image_length, connectivity_settings):
    pass    

def map_data(filtered_input, connectivity):
    pass

