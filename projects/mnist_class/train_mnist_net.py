import sys
import os
sys.path.insert(0, r"C:\Users\reidj\Dropbox\dphil\programming\spikes\spikes")
if __name__ == "__main__":
    print("Imports")
    from brian2 import *
    from network import *
    from input import *
    from tools import set_project_environment
    import numpy as np
    import matplotlib.pyplot as plt
    equations_container = EquationsContainer()
    network = Network()
    project = "mnist_class_wip"
    dir = os.path.dirname(os.path.abspath(__file__))
    set_project_environment(dir, project)
    # Set up parameters
    N_layers = 4
    STIMULUS_LENGTH = 100 * ms
    NUM_STIMULI = 10
    