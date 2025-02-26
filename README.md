## Spikes:

├── .venv
├── .windowsvenv
├── spikes
│   ├── data                        # to store various input data
│   │   ├── 3N2P
│   ├───├── mnist_fashion
│   │   └── random stuff
│   ├── input                       # To generate input from input data
│   │   ├── gabor_filters           # Where gabor filters are stored
│   │   ├── convolution.py          # Code which convolves images with filters
│   │   ├── gabor_filters.py        # Code which generates filters
│   │   └── mapping.py              # Code which turns filtered images into input matrices
│   ├── network                     # Making the Neural network
│   │   ├── create_network.py       # Combining specs to make the VisNet network
│   │   ├── equations.py            # Define equations used in code (note caching issues)
│   │   ├── neurons.py              # Define NeuronSpecs
│   │   └── synapses.py             # Define SynapseSpecs
│   └── results                     # Store Simulation Results
│       └── ...
│   ├── run                         # The running of the network
│   │   ├── files.py                # 
│   │   ├── monitors.py             # Setting up, managing, and displaying monitors
│   │   └── train_test.py           # Run test and train epochs
│   └── tests                       # to keep code working
│       └── tests.py
│   └── visualised_spikes           # Some monitors visualisation results
│       └── ...
├── main.py                         # run the main code
├── data_analysis.ipynb             # analyse spike results
├── docs                            # to document the functionality of the package
│   └── example.rst
├── README.md
├── NOTES.md
├── TODO.md