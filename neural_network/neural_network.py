from .layer import Layer
from .activations import ActivationLayer
from .activation_functions import get_activation_function

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, activation_function, loss_function,
                 loss_function_prime):
        self.layers = []
        self.loss = loss_function
        self.loss_prime = loss_function_prime

        activation, activation_prime = get_activation_function(activation_function)

        # Input Layer
        self.add_layer(Layer(input_size, neurons_per_layer))
        self.add_layer(ActivationLayer(activation, activation_prime))

        # Hidden Layers
        for _ in range(hidden_layers):
            self.add_layer(Layer(neurons_per_layer, neurons_per_layer))
            self.add_layer(ActivationLayer(activation, activation_prime))

        # Output Layer
        self.add_layer(Layer(neurons_per_layer, output_size))
        self.add_layer(ActivationLayer(activation, activation_prime))

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples

            if (i + 1) % 100 == 0:
                print(f'Epoch {i + 1}/{epochs}   Error={err}')
