from PySide6.QtCore import QThread, Signal

class TrainingThread(QThread):

    def __init__(self, neural_network, x_data, y_data, epochs, learning_rate):
        super().__init__()
        self.neural_network = neural_network
        self.x_data = x_data
        self.y_data = y_data
        self.epochs = epochs
        self.learning_rate = learning_rate

    def run(self):
        self.neural_network.fit(self.x_data, self.y_data, self.epochs, self.learning_rate)
