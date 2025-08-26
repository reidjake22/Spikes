
# Spikes: Investigating Polychronous Neural Groups in Spiking Neural Networks

Welcome to **Spikes**, an academic research project from the Oxford Centre for Theoretical Neuroscience and Artificial Intelligence. This project is dedicated to exploring the computational power and emergent dynamics of spiking neural networks, with a special focus on **polychronous neural groups** as described by Izhikevich (2008).

## Project Goals

**Spikes** aims to:
- Investigate the formation and function of polychronous groups—complex, time-locked firing patterns that may underlie memory and computation in the brain.
- extend the findings of Izhikevich (2008), using modern simulation tools and large-scale datasets.
- Develop unsupervised hierarchical spiking neural networks capable of learning robust features from noisy sensory input (e.g., MNIST images).
- Quantify information flow, feature extraction, and generalisation in spiking networks using rigorous statistical and information-theoretic methods.

## Academic Context

This project is part of ongoing research at Oxford, contributing to the understanding of how biological principles can inspire artificial intelligence. Polychronous groups are hypothesised to be a key mechanism for flexible computation and memory in the brain, and this codebase provides tools to simulate, analyse, and visualise their emergence in artificial networks.

## Code Structure

The repository is organised for clarity and extensibility:

```
spikes/
├── data/                # Input datasets (MNIST, fashion, etc.)
├── input/               # Input generation and preprocessing
│   ├── gabor_filters/   # Gabor filter storage
│   ├── convolution.py   # Image convolution routines
│   ├── gabor_filters.py # Filter generation
│   └── mapping.py       # Input matrix construction
├── network/             # Network construction and equations
│   ├── create_network.py
│   ├── equations.py
│   ├── neurons.py
│   └── synapses.py
├── results/             # Simulation outputs
├── run/                 # Training and testing scripts
│   ├── files.py
│   ├── monitors.py
│   └── train_test.py
├── tests/               # Unit tests
│   └── tests.py
├── visualised_spikes/   # Visualisation outputs
├── main.py              # Main entry point
├── data_analysis.ipynb  # Analysis notebook
├── docs/                # Documentation
│   └── example.rst
├── README.md            # This file
├── NOTES.md             # Research notes
├── TODO.md              # Project tasks
```

## Key Features

- **Unsupervised hierarchical SNNs**: Train on large datasets, record weights and spike counts at multiple checkpoints.
- **Noise robustness**: Test feature extraction and generalisation under varying input noise levels.
- **Information-theoretic analysis**: Quantify neuron/class information using specific information and related metrics.
- **Extensible framework**: Modular code for input generation, network construction, training, and analysis.

## Citation

If you use this codebase in your research, please cite:

> Izhikevich, E. M. (2008). Polychronization: Computation with spikes. Neural Computation, 18(2), 245-282.

## Contact

For questions or collaboration, contact the Oxford Centre for Theoretical Neuroscience and Artificial Intelligence.
