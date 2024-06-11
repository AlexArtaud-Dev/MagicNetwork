import sys
import logging

from IPython.external.qt_for_kernel import QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QFileDialog, QTableView, QMessageBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import numpy as np
import pandas as pd
from neural_network.neural_network import NeuralNetwork
from neural_network.activation_functions import ActivationFunction
from neural_network.loss_functions import mse, mse_prime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CSVData:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def load_csv(self, file_path, train=True):
        data = pd.read_csv(file_path)
        if train:
            self.x_train = data.iloc[:, :-1].values
            self.y_train = data.iloc[:, -1].values.reshape(-1, 1)
            logging.info(f'Training CSV loaded: {file_path}')
        else:
            self.x_train = data.values
            logging.info(f'Test CSV loaded: {file_path}')
        return data


class NeuralNetworkApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Creator")
        self.setGeometry(200, 200, 800, 600)

        self.csv_data = CSVData()
        self.network = None

        self.initUI()

    def initUI(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.activation_function_label = QLabel("Fonction d'activation :")
        self.activation_function_combo = QComboBox()
        self.activation_function_combo.addItems([func.name for func in ActivationFunction])
        self.activation_function_combo.currentIndexChanged.connect(self.update_activation_defaults)

        self.input_size_label = QLabel("Nombre de caractéristiques d'entrée :")
        self.input_size_spin = QSpinBox()
        self.input_size_spin.setMinimum(1)
        self.input_size_spin.setMaximum(100)
        self.input_size_spin.setReadOnly(True)

        self.output_size_label = QLabel("Nombre de caractéristiques de sortie :")
        self.output_size_spin = QSpinBox()
        self.output_size_spin.setMinimum(1)
        self.output_size_spin.setMaximum(100)
        self.output_size_spin.setReadOnly(True)

        self.hidden_layers_label = QLabel("Nombre de couches cachées :")
        self.hidden_layers_spin = QSpinBox()
        self.hidden_layers_spin.setMinimum(1)
        self.hidden_layers_spin.setMaximum(10)

        self.neurons_per_layer_label = QLabel("Nombre de neurones par couche :")
        self.neurons_per_layer_spin = QSpinBox()
        self.neurons_per_layer_spin.setMinimum(1)
        self.neurons_per_layer_spin.setMaximum(100)

        self.epochs_label = QLabel("Nombre d'itérations d'entraînement :")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(10000)

        self.learning_rate_label = QLabel("Taux d'apprentissage :")
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setMinimum(0.0001)
        self.learning_rate_spin.setMaximum(1.0)
        self.learning_rate_spin.setSingleStep(0.0001)

        self.activation_info_label = QLabel("")

        self.import_train_button = QPushButton("Importer CSV d'entraînement")
        self.import_train_button.clicked.connect(self.import_train_csv)

        self.train_button = QPushButton("Entraîner")
        self.train_button.clicked.connect(self.train_network)

        self.import_test_button = QPushButton("Importer CSV de test")
        self.import_test_button.clicked.connect(self.import_test_csv)

        self.predict_button = QPushButton("Prédire")
        self.predict_button.clicked.connect(self.predict)

        self.result_table = QTableView()

        layout.addWidget(self.activation_function_label)
        layout.addWidget(self.activation_function_combo)
        layout.addWidget(self.activation_info_label)
        layout.addWidget(self.input_size_label)
        layout.addWidget(self.input_size_spin)
        layout.addWidget(self.output_size_label)
        layout.addWidget(self.output_size_spin)
        layout.addWidget(self.hidden_layers_label)
        layout.addWidget(self.hidden_layers_spin)
        layout.addWidget(self.neurons_per_layer_label)
        layout.addWidget(self.neurons_per_layer_spin)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_spin)
        layout.addWidget(self.learning_rate_label)
        layout.addWidget(self.learning_rate_spin)
        layout.addWidget(self.import_train_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.import_test_button)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_table)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_activation_defaults(self):
        current_activation = self.activation_function_combo.currentText()
        logging.info(f'Activation function changed to: {current_activation}')
        info = {
            "RELU": "ReLU : Sortie [0, +∞), dérivée 1 si x > 0, sinon 0.",
            "TANH": "Tanh : Sortie [-1, 1], dérivée 1 - tanh(x)^2.",
            "SIGMOID": "Sigmoïde : Sortie (0, 1), dérivée σ(x)*(1 - σ(x)).",
            "LEAKY_RELU": "Leaky ReLU : Sortie (-∞, +∞), dérivée 1 si x > 0, sinon α (α=0.01)."
        }
        self.activation_info_label.setText(info[current_activation])

        if current_activation == "RELU":
            self.learning_rate_spin.setValue(0.01)
            self.epochs_spin.setValue(1000)
        elif current_activation == "TANH":
            self.learning_rate_spin.setValue(0.01)
            self.epochs_spin.setValue(1000)
        elif current_activation == "SIGMOID":
            self.learning_rate_spin.setValue(0.1)
            self.epochs_spin.setValue(5000)
        elif current_activation == "LEAKY_RELU":
            self.learning_rate_spin.setValue(0.01)
            self.epochs_spin.setValue(1000)

    def import_train_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Importer CSV d'entraînement", "", "CSV Files (*.csv)")
        if file_path:
            try:
                data = self.csv_data.load_csv(file_path, train=True)
                self.input_size_spin.setValue(self.csv_data.x_train.shape[1])
                self.output_size_spin.setValue(self.csv_data.y_train.shape[1])
                self.input_size_spin.setReadOnly(True)
                self.output_size_spin.setReadOnly(True)
                QMessageBox.information(self, "Succès", "CSV d'entraînement importé avec succès.")
            except Exception as e:
                logging.error(f'Error loading training CSV: {e}')
                QMessageBox.critical(self, "Erreur", f"Erreur lors de l'importation du CSV : {e}")

    def train_network(self):
        try:
            input_size = self.csv_data.x_train.shape[1]
            output_size = self.csv_data.y_train.shape[1]
            hidden_layers = self.hidden_layers_spin.value()
            neurons_per_layer = self.neurons_per_layer_spin.value()
            activation_function = ActivationFunction[self.activation_function_combo.currentText()]
            epochs = self.epochs_spin.value()
            learning_rate = self.learning_rate_spin.value()

            self.network = NeuralNetwork(
                input_size, output_size, hidden_layers, neurons_per_layer,
                activation_function, mse, mse_prime
            )
            self.network.fit(self.csv_data.x_train, self.csv_data.y_train, epochs=epochs, learning_rate=learning_rate)
            logging.info('Network training completed successfully.')
            QMessageBox.information(self, "Succès", "Réseau entraîné avec succès.")

            # Verrouiller les paramètres après l'entraînement
            self.hidden_layers_spin.setReadOnly(True)
            self.neurons_per_layer_spin.setReadOnly(True)
            self.epochs_spin.setReadOnly(True)
            self.learning_rate_spin.setReadOnly(True)
            self.activation_function_combo.setEnabled(False)
        except Exception as e:
            logging.error(f'Error during network training: {e}')
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'entraînement du réseau : {e}")

    def import_test_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Importer CSV de test", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.csv_data.load_csv(file_path, train=False)
                QMessageBox.information(self, "Succès", "CSV de test importé avec succès.")
            except Exception as e:
                logging.error(f'Error loading test CSV: {e}')
                QMessageBox.critical(self, "Erreur", f"Erreur lors de l'importation du CSV : {e}")

    def predict(self):
        try:
            predictions = self.network.predict(self.csv_data.x_train)
            predictions = np.array(predictions).flatten()

            result_df = pd.DataFrame(predictions, columns=["Prédictions"])
            self.result_table.setModel(DataFrameModel(result_df))
            logging.info('Predictions generated successfully.')
            QMessageBox.information(self, "Succès", "Prédictions générées avec succès.")
        except Exception as e:
            logging.error(f'Error during prediction: {e}')
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la prédiction : {e}")


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._df.columns[col]
        return None
