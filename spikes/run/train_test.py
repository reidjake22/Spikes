from brian2 import *
from .monitors import Monitors

# Making variables available here gets tricky if they're defined externally


def set_plasticity(network, enable):
    for obj in network.objects:
        if hasattr(obj, "plasticity"):
            obj.plasticity = enable
            print(f"plasticity set to {enable} for {obj.name}")


def run_training(network, namespace, stimulus_length, no_stimuli, no_epochs):
    epoch_length = stimulus_length * no_stimuli
    # for obj in network.objects:
    #     if isinstance(obj, Synapses):
    #         print("synapse object found:")
    #         if hasattr(obj, "plasticity"):
    #             print(f"{obj.name} plasticity: {obj.plasticity}")
    #         if hasattr(obj, "learning_rate"):
    #             print(f"{obj.name} learning_rate: {obj.learning_rate}")

    print(
        f"running {no_epochs} epochs for a total length of time of {stimulus_length * no_stimuli * no_epochs} ms"
    )
    for i in range(no_epochs):

        current_list = np.array([])
        for obj in network.objects:
            if isinstance(obj, Synapses):

                print("synapse object found:")
                if obj.name == "efe_2":
                    print(obj.w[:25])
                    print(obj.plasticity)
                    print(obj.learning_rate)
                    print(obj.apre)
                    print(obj.apost)
                    if i > 0:
                        changed_values = [
                            x for x, y in zip(obj.w, current_list) if x != y
                        ]
                        print(
                            f"changed_values (len={len(changed_values)})are: {changed_values}"
                        )
                    for i in obj.w:
                        current_list = np.append(current_list, i)
        print(f"running epoch no {i+1}")
        print(f"current time {network.t}")
        network.run(stimulus_length * no_stimuli, namespace=namespace)


def run_testing_epoch(monitors, network, namespace, stimulus_length, no_stimuli):
    epoch_length = stimulus_length * no_stimuli
    for obj in network.objects:
        if isinstance(obj, Synapses):
            print("synapse object found:")
            if hasattr(obj, "plasticity"):
                print(f"{obj.name} plasticity: {obj.plasticity}")
            if hasattr(obj, "learning_rate"):
                print(f"{obj.name} learning_rate: {obj.learning_rate}")
    # Need to turn off training
    monitors.toggle_monitoring(enable=True)
    set_plasticity(network, 0)
    print(f"current time {network.t}")
    network.run(stimulus_length * no_stimuli, namespace=namespace)
    print("Test Complete")
    print(f"current time {network.t}")


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
