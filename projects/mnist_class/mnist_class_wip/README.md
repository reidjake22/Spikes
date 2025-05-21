# Project: mnist_class_wip

Project description goes here.
So basically we want to run 60 epochs on the mnist dataset - this is gonna be hard as hell seeing as mnist is way way larger than anything else I'm used to 

The structure:
config has all the stuff we need to setup and run a network without major setup time (thanks setup_mnist_experimental.py).
    Weights/delays/connections
    mapping (helps us predetermine input)
    what "create_network" needs to do now is take these stored variables, load them in, create an input layer.
    Then main runner will handle arranging inputs & saving intermediary states (saving intermediary states should happen in config/<epochno>)

As I've mucked around with input a little we should first validate this, then look at training the network and visualising it as previously, then look at training at scale.