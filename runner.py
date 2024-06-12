from enum import Enum
import numpy as np

from gui.main_window import NeuralNetworkApp
from neural_network.csv_dataloader import CSVDataLoader
from neural_network.neural_network import NeuralNetwork
from neural_network.activation_functions import ActivationFunction
from neural_network.loss_functions import mse, mse_prime

class Runner(Enum):
    CLI = 1
    GUI = 2
    TEST = 3

def execute_runner(runner):
    if runner == Runner.CLI:
        cli()
    elif runner == Runner.GUI:
        gui()
    elif runner == Runner.TEST:
        test()
    else:
        raise ValueError("Unsupported runner")

def cli():
    # Données XOR
    x_train = np.array([
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]]
    ])
    y_train = np.array([[0], [1], [1], [0]])

    print("X data:", x_train)
    print("Y data:", y_train)

    input_size = 2
    output_size = 1
    hidden_layers = 1  # Une couche cachée est souvent suffisante pour XOR
    neurons_per_layer = 2  # Deux neurones dans la couche cachée
    activation_function = ActivationFunction.LEAKY_RELU  # Utiliser Tanh pour le problème XOR

    net = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer, activation_function, mse, mse_prime)
    net.fit(x_train, y_train, epochs=100000, learning_rate=0.1)

    out = net.predict(x_train)
    print(out)

def test():
    file_path = 'xor_train.csv'
    input_size = 2  # Example input size
    output_size = 1  # Example output size

    data_loader = CSVDataLoader(file_path, input_size, output_size)
    data_loader.load_and_check_data()
    x_data, y_data = data_loader.prepare_data()

    print("X data:", x_data)
    print("Y data:", y_data)

    input_size = 2
    output_size = 1
    hidden_layers = 1  # Une couche cachée est souvent suffisante pour XOR
    neurons_per_layer = 2  # Deux neurones dans la couche cachée
    activation_function = ActivationFunction.TANH  # Utiliser Tanh pour le problème XOR

    net = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer, activation_function, mse, mse_prime)
    net.fit(x_data, y_data, epochs=50000, learning_rate=0.1)

    out = net.predict(x_data)
    print(out)

def gui():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    ex = NeuralNetworkApp()
    ex.show()
    sys.exit(app.exec())
