from brian2 import *


def describe_network_components(network):
    """
    Prints a detailed description of the NeuronGroups and Synapses in a Brian2 Network.

    Parameters:
        network (Network): The Brian2 Network object to describe.
    """
    print("===== Network Components =====\n")

    # Header for NeuronGroups
    print("NeuronGroups".center(80, "="))
    print(f"{'Name':<20}{'Neuron Count':<15}{'Parameters':<25}{'Model':<20}")
    print("=" * 80)

    for obj in network.objects:
        if isinstance(obj, NeuronGroup):
            # Extract name, neuron count, and model
            name = obj.name
            neuron_count = obj.N
            model = str(obj.equations)

            # Safely handle parameters, excluding auxiliary variables
            parameters = {}
            for param in obj.variables.keys():
                variable = obj.variables[param]
                if variable.read_only:  # Skip auxiliary variables like xi
                    continue
                try:
                    value = getattr(obj, param)
                    parameters[param] = (
                        value[:] if hasattr(value, "__getitem__") else value
                    )
                except Exception:
                    parameters[param] = None

            # Format parameters as key-value pairs
            parameters_str = ", ".join(
                [
                    f"{k}={v.mean():.3g}" if hasattr(v, "mean") else f"{k}={v}"
                    for k, v in parameters.items()
                ]
            )
            print(f"{name:<20}{neuron_count:<15}{parameters_str:<25}{model:<20}")
    print("\n")

    # Header for Synapses
    print("Synapses".center(80, "="))
    print(
        f"{'Name':<20}{'Source -> Target':<20}{'Synapse Count':<15}{'Parameters':<25}{'Model':<20}"
    )
    print("=" * 80)

    for obj in network.objects:
        if isinstance(obj, Synapses):
            # Extract name, connection details, and model
            name = obj.name
            connection = f"{obj.source.name} -> {obj.target.name}"
            synapse_count = len(obj)
            model = str(obj.equations)

            # Safely handle parameters, excluding auxiliary variables
            parameters = {}
            for param in obj.variables.keys():
                variable = obj.variables[param]
                if variable.read_only:  # Skip auxiliary variables like xi
                    continue
                try:
                    value = getattr(obj, param)
                    parameters[param] = (
                        value[:] if hasattr(value, "__getitem__") else value
                    )
                except Exception:
                    parameters[param] = None

            # Format parameters as key-value pairs
            parameters_str = ", ".join(
                [
                    f"{k}={v.mean():.3g}" if hasattr(v, "mean") else f"{k}={v}"
                    for k, v in parameters.items()
                ]
            )
            print(
                f"{name:<20}{connection:<20}{synapse_count:<15}{parameters_str:<25}{model:<20}"
            )
    print("\n")


def determine_inputs_per_image(data, flat_poisson_inputs, beta=6, num_neurons=4096):
    # The shape of flat_poisson_inputs is num_images, num_poisson_neurons
    # Create directory if it doesn't exist
    num_images, num_poisson_neurons = flat_poisson_inputs.shape
    print(f"num_images: {num_images}, num_poisson_neurons: {num_poisson_neurons}")
    collapsed_input_hz = flat_poisson_inputs * beta * hertz
    # shape: num_images by (height x width x num_filters)
    i_indices = data["i_0"]
    print(len(i_indices))
    j_indices = data["j_0"]
    print(max(j_indices))
    j_activity = np.zeros((num_images, num_neurons)) * hertz
    for image in range(num_images):
        print(f"image: {image}")
        for i, j in zip(i_indices, j_indices):
            if collapsed_input_hz[image, i] > 0:
                j_activity[image, j] += collapsed_input_hz[image, i]
    print(j_activity[:, 0])
    return j_activity
