import numpy as np
from neural_network.neuron import Neuron
from neural_network.activation_functions import get_activation_function

class NeuralLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.activation, self.activation_prime = get_activation_function(activation_function)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.hstack([neuron.forward(input_data) for neuron in self.neurons])
        self.output = self.activation(self.output)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        output_error = output_error * self.activation_prime(self.output)
        input_error = np.zeros_like(self.input, dtype=np.float64)
        output_error = output_error.T
        for i, neuron in enumerate(self.neurons):
            input_error += neuron.backward(output_error[i].reshape(-1, 1), learning_rate)
        return input_error
