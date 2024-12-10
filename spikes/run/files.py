import orjson
import os


def store_synapses(
    storage, n_layers, exc_neuron_specs, inh_neuron_specs, directory, file_name
):
    print("storing data")
    storage["help"] = (
        f"this is a set of weights for {n_layers} layers,with format <efferent: e or i><type: f, l, b><layer: 0, 1, 2, 3> for the keys.The number of excitatory neurons are {exc_neuron_specs.length} and the number of inhibitory neurons are {inh_neuron_specs.length}"
    )
    file_path = os.path.join(directory, file_name)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, "wb") as file:
        file.write(orjson.dumps(storage))
    print(f"data stored in {file_path}")
    try:
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())
        print(data["efe_1"][0])
        print("data loaded")
    except:
        print("data not loaded")


def load_synapses(directory, file_name):
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        return None
    try:
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())
        print(f"data loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None
