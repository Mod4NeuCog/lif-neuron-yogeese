import numpy as np

"""
Problem
1. make a neuron class with abstract funcion (sigmoid, leaky-integ fire, etc. )
2. make 2 layers -> network (store weights in a matrix)
  1. from the last (second layer) take the max out of the layer for the classification
3. make a hidden layer to make 3 layers in total
"""


class Neuron:
    def __init__(self, input=0):
        self.input = input
        self.voltage_membrane = 0
        self.activation = 0
        self.voltage_threshold = 10

    def sigmoid(self, input):
        output = np.exp(input)/(1.0 + np.exp(input))
        return output

    def integrate_fire(self,input):
        """
        this type of neuron accumulates the inputs and 
        stores in membrane potential.
        the neuron fires if it is over the threshold Vth
        """
        self.input = input
        self.voltage_membrane = 
        if self.voltage_membrane > self.voltage_threshold:
            output = "make a spike"  # TODO
            self.activation = output
            return self.activation

    def output(self):
        self.voltage_membrane = self.input
        self.activation = self.sigmoid(self.voltage_membrane)
        return self.activation


# neuron1 = Neuron(1)
# neuron1.output()
# neuron1.input = 2
# neuron1.output()


class Layer(Neuron):
    def __init__(self, n_neurons, input=None):
        self.n_neurons = n_neurons
        self.layer = []
        self.generate_layer()

    def generate_layer(self):
        self.layer = []
        for i in range(self.n_neurons):
            self.layer.append(Neuron(0))
        return self.layer

    def input(self, neuron_input):
        self.inputs = []
        assert len(
            neuron_input) == self.n_neurons, "input vector should be the same as the number of neurons"
        for i, neuron in enumerate(self.layer):
            neuron.input = neuron_input[i]
            self.inputs.append(neuron.input)

    def activate(self):
        self.activations = []
        for neuron in self.layer:
            self.activations.append(neuron.output())


layer1 = Layer(n_neurons=2)
layer1.layer
layer1.n_neurons
layer1.generate_layer()
layer1.input([1, 2])
layer1.activate()
layer1.activations


class Network(Layer):
    def __init__(self, input_neurons, output_neurons, hidden_layers_spec=[]):
        self.layers = []
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_layers_spec = hidden_layers_spec
        self.hidden_layers = []
        self.weights = {}
        self.generate_layer()
        self.generate_weights()

    def generate_layer(self):
        input_layer = Layer(self.input_neurons)
        output_layer = Layer(self.output_neurons)
        if len(self.hidden_layers_spec) != 0:
            self.hidden_layers = []
            for neurons_hidden_layer in self.hidden_layers_spec:
                hidden_layer = Layer(neurons_hidden_layer)
                self.hidden_layers.append(hidden_layer)
            self.layers = [input_layer] + self.hidden_layers + [output_layer]
        else:
            self.layers = [input_layer, output_layer]

    def generate_weights(self):
        # get the layer's dimensions and make a matrix of random numbers
        for i, layer in enumerate(self.layers):
            if i+1 < len(self.layers):
                next_layer = self.layers[i+1]
                self.weights['weight_layer_'+str(i)+str(i+1)] = np.ones([
                    layer.n_neurons, next_layer.n_neurons])

    def summary(self):
        print("Layer:            : ", "No of Neurons\n")
        print("Input Layer       : ", len(self.layers[0].layer), "\n")
        for i, layer in enumerate(self.hidden_layers):
            print("Hidden Layer[", i, "] : ", len(layer.layer), "\n")
        print("Output Layer      : ", len(self.layers[-1].layer), "\n")

    def forward_pass(self, input):
        self.layers[0].input(input)
        for i, layer in enumerate(self.layers):
            if i+1 < len(self.layers):
                layer.activate()
                act = layer.activations
                weight_matrix = self.weights['weight_layer_'+str(i)+str(i+1)]
                next_input = np.dot(np.transpose(weight_matrix), act)
                # next_input = np.dot(act, weight_matrix)
                self.layers[i+1].input(next_input)
        self.layers[-1].activate()
        return np.argmax(self.layers[-1].activations)


network = Network(2, 2)
network.summary()
network.forward_pass([1,0])
network.layers[-1].activations
network.layers[0].activations
