import math
import numpy as np

"""
Question
1. make a neuron class with abstract funcion (sigmoid, leaky-integ fire, etc. )
2. make 2 layers -> network (store weights in a matrix)
  1. from the last (second layer) take the max out of the layer for the classification
"""

class Neuron:
    def __init__(self, input, activation):
        self.input = input
        self.activation = activation

    def sigmoid(self, input):
        output = np.exp(input)/(1.0 + np.exp(input))
        return output

    def output(self):
        return self.sigmoid(input)

class Layer(Neuron):
    """
    n_neurons: number of neurons
    """
    def __init__(self, n_neurons, function, input):
        self.n_neurons = n_neurons
        self.function = function
        self.input = input
        self.neurons = np.array([])

    def make_layer():
        for i in n_neurons:
            ith_neuron = Neuron(self.input, self.function)
            self.neurons.append()
        return self.neurons

class Model(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.matrix([])
        self.input_size = input_size
        self.output_size = output_size

    def make_input_output_layers(self.input_size, self.output_size):
        pass

    def add_hidden_layer(n_neurons):
        pass

    def forward(input):




# class Network(layers, weights):
    # pass

# def Neuron(input, function):
#     return function(input)

def neuronlayer():
    pass 

def connect(neuronlayerlist):
    pass 


def lif(input):
    Vthreshold = 70
    pass

example = Neuron( -1, sigmoid )
