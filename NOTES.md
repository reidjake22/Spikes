working with the venv:
To activate::
source .venv/bin/activate
To deactivate:
deactivate

Running code:
Python3 -m spikes/main

So on the filter module:
    the flow is as follows
    generate the gabor filters according to our specs
    then when you have your data set loaded in as a 3D array convolve each item with the gabor filters.
    The result of each convolved image should be a 3D array, let's store all the images as a 4D array and make this process of convolving a 3D array with the filters to create a 4D array a function. Next I want to take this 4D array of gabor filters and generate a neuron input according to a prespecified mapping between neurons of the layer and the gabor filter. This is using the radius stuff mentioned earlier. Ma