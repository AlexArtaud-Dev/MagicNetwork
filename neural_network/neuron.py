import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size, 1) * 0.1  # Shape (input_size, 1)
        self.bias = np.zeros((1, 1))  # Ensure bias has shape (1, 1)
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error.reshape(self.weights.shape)  # Ensure shapes match
        self.bias -= learning_rate * output_error.reshape(self.bias.shape)  # Ensure shapes match
        return input_error