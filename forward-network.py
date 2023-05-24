import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")

"""
Problem
1. make a neuron class with abstract funcion (sigmoid, leaky-integ fire, etc. )
2. make 2 layers -> network (store weights in a matrix)
  1. from the last (second layer) take the max out of the layer for the classification
3. make a hidden layer to make 3 layers in total
"""


class Neuron:
    def __init__(self, n_type="sigmoid",
                 threshold=-65, reset=-, resting=-75, tau=10e-3):
        # self.input = input
        self.activation = 0
        self.voltage_membrane = resting
        self.voltage_reset = reset
        self.voltage_threshold = threshold
        self.n_type = n_type
        self.tau = tau

    def sigmoid(self, input):
        output = np.exp(input)/(1.0 + np.exp(input))
        self.activation = output
        return self.activation

    def integrate_fire(self, input, time_step=1e-3):
        """
        this type of neuron accumulates the inputs and 
        stores in membrane potential.
        the neuron fires if it is over the threshold Vth
        """
        # the given input here should be already be the total input (spikes) multiplied by the weights
        beta = np.exp(-time_step/self.tau)  # the leak term
        self.voltage_membrane = (beta*self.voltage_membrane) + input
        if self.voltage_membrane >= self.voltage_threshold:
            output = 1
            self.activation = output
            self.voltage_membrane = self.voltage_reset
            return self.activation
        else:
            self.activation = 0
            return self.activation

    def output(self, input, time_step=1e-3):
        if self.n_type == "sigmoid":
            self.sigmoid(input)
            return self.activation
        if self.n_type == "lif":
            self.integrate_fire(input, time_step=time_step)
            return self.activation


# test the sigmoid neuron
neuron0 = Neuron(n_type="sigmoid")
neuron0.output(0)
neuron0.output(2)

# test the lif inspired neuron
time_step = 1e-2
n_steps = 1000

time_vector = np.arange(-10, 10, time_step)
len(time_vector)
input_vector = 5 + np.sin(time_vector*0.5)
# input_vector = np.heaviside(time_vector,1)


def stepfn(start, stop, amplitude):
    input_vector = np.zeros_like(time_vector)
    for i, t in enumerate(time_vector):
        if start < t < stop:
            input_vector[i] = amplitude
    return input_vector

# input_vector = stepfn(-5,5,100)


neuron1 = Neuron(n_type="lif")
activity = np.array([])
spikes = np.array([])
for i in input_vector:
    output = neuron1.output(i, time_step=time_step)
    spikes = np.append(spikes, output)
    activity = np.append(activity, neuron1.voltage_membrane)

np.unique(spikes)
spike_index = []
for i, a in enumerate(spikes):
    if a == 1:
        spike_index.append(i)
spike_index
print(activity)
input_vector

plt.figure()
plt.plot(time_vector, np.c_[activity, spikes, input_vector],
         label=['membrane vol', 'spikes', 'input'])
plt.legend()
plt.show()

# now I need to rewirte the layer code to simulate the neurons for a number of time steps
# should I do this in the network instead?


class Layer(Neuron):
    def __init__(self, n_neurons, n_type="sigmoid", input=None):
        self.n_neurons = n_neurons
        self.n_type = n_type
        self.layer = []
        self.generate_layer()

    def generate_layer(self):
        self.layer = []
        for i in range(self.n_neurons):
            # self.layer.append(Neuron())
            self.layer.append(Neuron(n_type="lif"))
        return self.layer

    def input(self, neuron_input):
        self.inputs = []
        assert len(
            neuron_input) == self.n_neurons, "input vector should be the same as the number of neurons"
        for i, neuron in enumerate(self.layer):
            neuron.input = neuron_input[i]
            self.inputs.append(neuron.input)

    def activate(self, input):
        assert len(
            input) == self.n_neurons, "input vector should be the same as the number of neurons"
        self.activations = []
        for i, neuron in enumerate(self.layer):
            self.activations.append(neuron.output(input[i]))


layer1 = Layer(n_neurons=2)
layer1.layer
layer1.n_neurons
layer1.generate_layer()
layer1.activate([20,2])
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
network.forward_pass([1, 0])
network.layers[-1].activations
network.layers[0].activations
