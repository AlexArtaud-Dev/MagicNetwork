from PySide6.QtCore import QThread

from neural_network.neural_network import NeuralNetwork


class TrainingThread(QThread):

    def __init__(self, neural_network: NeuralNetwork, x_data, y_data, epochs: int, learning_rate: int) -> None:
        super().__init__()
        self.neural_network = neural_network
        self.x_data = x_data
        self.y_data = y_data
        self.epochs = epochs
        self.learning_rate = learning_rate

    def run(self) -> None:
        self.neural_network.fit(self.x_data, self.y_data, self.epochs, self.learning_rate)
