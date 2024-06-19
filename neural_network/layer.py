from neural_network.neuron import Neuron
import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.hstack([neuron.forward(input_data) for neuron in self.neurons])
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros_like(self.input, dtype=np.float64)  # Ensure float64 type
        output_error = output_error.T  # Transpose to match neuron iteration
        for i, neuron in enumerate(self.neurons):
            input_error += neuron.backward(output_error[i].reshape(-1, 1), learning_rate)
        return input_error