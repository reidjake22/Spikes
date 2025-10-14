from brian2 import *

# Making variables available here gets tricky if they're defined externally


def set_plasticity(network, enable):
    for obj in network.objects:
        if hasattr(obj, "plasticity"):
            obj.plasticity = enable
            print(f"plasticity set to {enable} for {obj.name}")


def get_values_from_synapse(network, attributes):
    for obj in network.objects:
        if isinstance(obj, Synapses):
            dict_of_values = {}
            for i in attributes:
                try:
                    dict_of_values[str(i)] = obj.i
                except:
                    print(f"{obj.name} can't find {str(i)}")
            return dict_of_values


def run_training(network, namespace, stimulus_length, no_stimuli, no_epochs):
    epoch_length = stimulus_length * no_stimuli
    print(f"current time {network.t}")
    print(
        f"running {no_epochs} epochs for a total length of time of {stimulus_length * no_stimuli * no_epochs} ms"
    )
    for epoch_no in range(no_epochs):
        try:
            for obj in network.objects:
                print(f"Synapse info:")
                try:
                    plasticity = np.mean(obj.plasticity)
                except:
                    plasticity = "N/A"
                try:
                    learning_rate = np.mean(obj.learning_rate)
                except:
                    learning_rate = "N/A"
                try:
                    ga = np.mean(obj.ga)
                except:
                    ga = "N/A"
                if isinstance(obj, Synapses):
                    print(f"{'Name':<20} {'Plasticity':<15} {'Learning Rate':<15} {'GA':<15}")
                    print(f"{obj.name:<20} {plasticity:<15} {learning_rate:<15} {ga:<15}")
        except:
            print("Error getting synapse info")

        print(f"running epoch no {epoch_no+1}")
        print(f"current time {network.t}")
        network.run(stimulus_length * no_stimuli, namespace=namespace)


def run_testing_epochs(network, namespace, stimulus_length, no_stimuli, no_testing_epochs):
    epoch_length = stimulus_length * no_stimuli
    for obj in network.objects:
        if isinstance(obj, Synapses):
            print("synapse object found:")
            if hasattr(obj, "plasticity"):
                print(f"{obj.name} plasticity: {obj.plasticity}")
            if hasattr(obj, "learning_rate"):
                print(f"{obj.name} learning_rate: {obj.learning_rate}")

    set_plasticity(network, 0)
    for obj in network.objects:
        if isinstance(obj, Synapses):
            if hasattr(obj, "plasticity"):
                print(f"Plasticity is set to {obj.plasticity}")
    for epoch in range(no_testing_epochs):
        print(f"running testing epoch {epoch+1}")
        print(f"current time {network.t}")
        network.run(stimulus_length * no_stimuli, namespace=namespace)
        print(f"current time {network.t}")
    print("testing complete")

def toggle_plasticity(state):
    if not isinstance(state, bool):
        raise ValueError("State must be a boolean")
    global on_plasticity
    on_plasticity = state
    print(on_plasticity)


running_network = r"""

**********************************************************************
*                                                                    *
*   █████   ██    ██  ███    ██  ███    ██ █████ ███    ██  ██████   *
*   ██   ██ ██    ██  ████   ██  ████   ██  ██   ████   ██ ██        *
*   █████   ██    ██  ██ ██  ██  ██ ██  ██  ██   ██ ██  ██ ██    ██  *
*   ██   ██ ██    ██  ██  ██ ██  ██  ██ ██  ██   ██  ██ ██ ██     █  *
*   ██   ██  ██████   ██   ████  ██   ████ █████ ██   ████   █████   *
*                                                                    *
*                                                                    *
*   ██     ██ ███████  ██████ ██     ██   █████  █████    ██    ██   *
*   ████   ██ ██         ██   ██     ██  ██   ██ ██   ██  ██  ██     *
*   ██ ██  ██ █████      ██   ██  █  ██  ██   ██ █████    ████       *
*   ██  ██ ██ ██         ██   ██ ███ ██  ██   ██ ██   ██  ██  ██     *
*   ██   ████ ███████    ██    ███ ███    █████  ██   ██  ██    ██   *
*                                                                    *
*             ✨   BROUGHT TO YOU BY OCTNAI    ✨                     *
**********************************************************************
"""
