from neural_network.neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, activation_function, loss_function, loss_function_prime):
        self.layers = []
        self.loss = loss_function
        self.loss_prime = loss_function_prime

        # Input Layer
        self.add_layer(NeuralLayer(input_size, neurons_per_layer, activation_function))

        # Hidden Layers
        for _ in range(hidden_layers):
            self.add_layer(NeuralLayer(neurons_per_layer, neurons_per_layer, activation_function))

        # Output Layer
        self.add_layer(NeuralLayer(neurons_per_layer, output_size, activation_function))

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

    def fit(self, x_train, y_train, epochs=20000, learning_rate=0.1):
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
                print(f'Epoch {i + 1}/{epochs} |  Error= {err}')
