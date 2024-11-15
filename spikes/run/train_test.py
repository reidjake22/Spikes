from brian2 import *
from .monitors import Monitors

# Making variables available here gets tricky if they're defined externally


def run_training_epoch(network, namespace, stimulus_length, no_stimuli):
    network.run(stimulus_length * no_stimuli, namespace=namespace)


def set_plasticity(network, enable):
    for obj in network.objects:
        if hasattr(obj, "plasticity"):
            obj.plasticity = enable
            print(f"plasticity set to {enable} for {obj.name}")


def run_training(network, namespace, stimulus_length, no_stimuli, no_epochs):
    epoch_length = stimulus_length * no_stimuli
    print(
        f"running {no_epochs} epochs for a total length of time of {stimulus_length * no_stimuli * no_epochs} ms"
    )
    for i in range(no_epochs):
        print(f"running epoch no {i+1}")
        print(f"current time {network.t}")
        run_training_epoch(network, namespace, stimulus_length, no_stimuli)


def run_testing_epoch(monitors, network, namespace, stimulus_length, no_stimuli):
    epoch_length = stimulus_length * no_stimuli
    # Need to turn off training
    monitors.toggle_monitoring(enable=True)
    set_plasticity(network, 0)
    print(f"current time {network.t}")
    network.run(stimulus_length * no_stimuli, namespace=namespace)
    print("Test Complete")
    print(f"current time {network.t}")
