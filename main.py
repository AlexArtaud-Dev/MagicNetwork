import numpy as np
from neural_network.neural_network import NeuralNetwork
from neural_network.activation_functions import ActivationFunction
from neural_network.loss_functions import mse, mse_prime

if __name__ == "__main__":
    # Données XOR
    x_train = np.array([
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]]
    ])
    y_train = np.array([[0.0], [1.0], [1.0], [0.0]])

    input_size = 2
    output_size = 1
    hidden_layers = 1  # Une couche cachée est souvent suffisante pour XOR
    neurons_per_layer = 2  # Deux neurones dans la couche cachée
    activation_function = ActivationFunction.LEAKY_RELU  # Utiliser Tanh pour le problème XOR

    net = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer, activation_function, mse, mse_prime)
    net.fit(x_train, y_train, epochs=10000, learning_rate=0.01)

    out = net.predict(x_train)
    print(out)
