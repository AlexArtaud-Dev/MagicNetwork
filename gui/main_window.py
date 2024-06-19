import sys
import pandas as pd
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QSpinBox, QPushButton,
                               QFileDialog, QComboBox, QTextEdit, QHBoxLayout, QDoubleSpinBox, QProgressDialog,
                               QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPen, QBrush, QColor
from .training_thread import TrainingThread
from neural_network.csv_dataloader import CSVDataLoader
from neural_network.neural_network import NeuralNetwork
from neural_network.activation_functions import ActivationFunction
from neural_network.loss_functions import mse, mse_prime

class NeuralNetworkApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.showFullScreen()
        self.neural_network = None
        self.data_loader = None
        self.training_thread = None
        self.training_dialog = None

    def initUI(self):
        self.setWindowTitle("Neural Network Configurator")

        self.layout = QVBoxLayout()

        # Left side layout
        self.left_layout = QVBoxLayout()

        # Input size
        self.input_size_label = QLabel("Input Size:")
        self.input_size_spinbox = QSpinBox()
        self.input_size_spinbox.setRange(1, 100)
        self.input_size_spinbox.valueChanged.connect(self.update_visualization)
        self.left_layout.addWidget(self.input_size_label)
        self.left_layout.addWidget(self.input_size_spinbox)

        # Output size
        self.output_size_label = QLabel("Output Size:")
        self.output_size_spinbox = QSpinBox()
        self.output_size_spinbox.setRange(1, 100)
        self.output_size_spinbox.valueChanged.connect(self.update_visualization)
        self.left_layout.addWidget(self.output_size_label)
        self.left_layout.addWidget(self.output_size_spinbox)

        # Hidden layers
        self.hidden_layers_label = QLabel("Hidden Layers:")
        self.hidden_layers_spinbox = QSpinBox()
        self.hidden_layers_spinbox.setRange(1, 10)
        self.hidden_layers_spinbox.valueChanged.connect(self.update_visualization)
        self.left_layout.addWidget(self.hidden_layers_label)
        self.left_layout.addWidget(self.hidden_layers_spinbox)

        # Neurons per layer
        self.neurons_per_layer_label = QLabel("Neurons per Layer:")
        self.neurons_per_layer_spinbox = QSpinBox()
        self.neurons_per_layer_spinbox.setRange(1, 20)
        self.neurons_per_layer_spinbox.valueChanged.connect(self.update_visualization)
        self.left_layout.addWidget(self.neurons_per_layer_label)
        self.left_layout.addWidget(self.neurons_per_layer_spinbox)

        # Activation function
        self.activation_function_label = QLabel("Activation Function:")
        self.activation_function_combobox = QComboBox()
        self.activation_function_combobox.addItems([e.name for e in ActivationFunction])
        self.activation_function_combobox.currentIndexChanged.connect(self.update_activation_function_info)
        self.activation_function_combobox.currentIndexChanged.connect(self.update_visualization)
        self.left_layout.addWidget(self.activation_function_label)
        self.left_layout.addWidget(self.activation_function_combobox)

        # Activation function info
        self.activation_function_info = QLabel()
        self.left_layout.addWidget(self.activation_function_info)

        # Loss function
        self.loss_function_label = QLabel("Loss Function: MSE")
        self.left_layout.addWidget(self.loss_function_label)

        # Epochs
        self.epochs_label = QLabel("Epochs:")
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(10000, 100000)
        self.epochs_spinbox.setValue(10000)
        self.left_layout.addWidget(self.epochs_label)
        self.left_layout.addWidget(self.epochs_spinbox)

        # Learning rate
        self.learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(0.01, 1.0)
        self.learning_rate_spinbox.setValue(0.01)
        self.learning_rate_spinbox.setSingleStep(0.01)
        self.left_layout.addWidget(self.learning_rate_label)
        self.left_layout.addWidget(self.learning_rate_spinbox)

        # Load CSV button
        self.load_csv_layout = QHBoxLayout()
        self.load_csv_button = QPushButton("Load Training CSV")
        self.load_csv_button.clicked.connect(self.load_csv)
        self.load_csv_layout.addWidget(self.load_csv_button)
        self.left_layout.addLayout(self.load_csv_layout)

        # Train button
        self.train_button = QPushButton("Train Network")
        self.train_button.clicked.connect(self.train_network)
        self.left_layout.addWidget(self.train_button)

        # Status box
        self.status_box = QLabel("Status: Not Trained")
        self.status_box.setStyleSheet("color: orange")
        self.left_layout.addWidget(self.status_box)

        # Load CSV for prediction button
        self.load_csv_predict_button = QPushButton("Load Prediction CSV")
        self.load_csv_predict_button.setEnabled(False)
        self.load_csv_predict_button.clicked.connect(self.load_csv_for_prediction)
        self.left_layout.addWidget(self.load_csv_predict_button)

        # Prediction output
        self.prediction_output = QTextEdit()
        self.prediction_output.setReadOnly(True)
        self.left_layout.addWidget(self.prediction_output)

        # Right side layout for visualization
        self.right_layout = QVBoxLayout()
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setBackgroundBrush(QBrush(QColor(Qt.white)))
        self.graphics_view.setScene(self.graphics_scene)
        self.right_layout.addWidget(self.graphics_view)

        # Main layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        self.setLayout(self.main_layout)

    def update_activation_function_info(self):
        activation_function = self.activation_function_combobox.currentText()
        if activation_function == "RELU":
            self.activation_function_info.setText("ReLU: Output [0, +∞), Derivative 1 if x > 0 else 0")
            self.set_default_parameters(learning_rate=0.01, epochs=10000)
        elif activation_function == "TANH":
            self.activation_function_info.setText("Tanh: Output [-1, 1], Derivative 1 - tanh(x)^2")
            self.set_default_parameters(learning_rate=0.01, epochs=50000)
        elif activation_function == "SIGMOID":
            self.activation_function_info.setText("Sigmoid: Output (0, 1), Derivative σ(x)*(1-σ(x))")
            self.set_default_parameters(learning_rate=0.1, epochs=20000)
        elif activation_function == "LEAKY_RELU":
            self.activation_function_info.setText(
                "Leaky ReLU: Output mainly [0, +∞), Derivative 1 if x > 0 else α (α=0.01)")
            self.set_default_parameters(learning_rate=0.01, epochs=100000)

    def set_default_parameters(self, learning_rate, epochs):
        self.learning_rate_spinbox.setValue(learning_rate)
        self.epochs_spinbox.setValue(epochs)

    def update_visualization(self):
        input_size = self.input_size_spinbox.value()
        output_size = self.output_size_spinbox.value()
        hidden_layers = self.hidden_layers_spinbox.value()
        neurons_per_layer = self.neurons_per_layer_spinbox.value()

        self.graphics_scene.clear()

        layer_gap = 100
        neuron_gap = 50
        neuron_radius = 10

        layers = [input_size] + [neurons_per_layer] * hidden_layers + [output_size]

        # Calculate vertical offset to center the neurons
        max_neurons = max(layers)
        total_height = (max_neurons - 1) * neuron_gap + 2 * neuron_radius
        scene_height = self.graphics_scene.sceneRect().height()
        vertical_offset = (scene_height - total_height) / 2 if scene_height > total_height else 0

        for layer_index, neuron_count in enumerate(layers):
            for neuron_index in range(neuron_count):
                x = layer_index * layer_gap
                y = vertical_offset + neuron_index * neuron_gap

                neuron = QGraphicsEllipseItem(x, y, neuron_radius * 2, neuron_radius * 2)
                neuron.setBrush(QBrush(Qt.blue))
                self.graphics_scene.addItem(neuron)

                if layer_index > 0:
                    for prev_neuron_index in range(layers[layer_index - 1]):
                        prev_x = (layer_index - 1) * layer_gap
                        prev_y = vertical_offset + prev_neuron_index * neuron_gap

                        line = QGraphicsLineItem(prev_x + neuron_radius, prev_y + neuron_radius,
                                                 x + neuron_radius, y + neuron_radius)
                        line.setPen(QPen(Qt.black))
                        self.graphics_scene.addItem(line)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                data = pd.read_csv(file_path)
                input_size = data.filter(like='Input').shape[1]
                output_size = data.filter(like='Output').shape[1]
                self.input_size_spinbox.setValue(input_size)
                self.output_size_spinbox.setValue(output_size)

                self.data_loader = CSVDataLoader(file_path, input_size, output_size)
                self.data_loader.load_and_check_data()
                self.x_data, self.y_data = self.data_loader.prepare_data()

                self.load_csv_button.setText(file_path.split('/')[-1])
                self.remove_csv_button = QPushButton("Remove")
                self.remove_csv_button.clicked.connect(self.remove_csv)
                self.load_csv_layout.addWidget(self.remove_csv_button)
                self.load_csv_button.setEnabled(False)
                self.status_box.setText(f"Loaded CSV: {file_path.split('/')[-1]}")
                self.update_visualization()
            except Exception as e:
                self.status_box.setText(f"Error loading CSV: {e}")

    def remove_csv(self):
        self.load_csv_layout.removeWidget(self.load_csv_button)
        self.load_csv_layout.removeWidget(self.remove_csv_button)
        self.load_csv_button.deleteLater()
        self.remove_csv_button.deleteLater()
        self.load_csv_button = QPushButton("Load Training CSV")
        self.load_csv_button.clicked.connect(self.load_csv)
        self.load_csv_layout.addWidget(self.load_csv_button)
        self.load_csv_button.setEnabled(True)
        self.data_loader = None
        self.x_data = None
        self.y_data = None
        self.load_csv_predict_button.setEnabled(False)
        self.status_box.setText("Status: Not Trained")
        self.status_box.setStyleSheet("color: orange")
        self.graphics_scene.clear()

    def train_network(self):
        if self.data_loader:
            input_size = self.input_size_spinbox.value()
            output_size = self.output_size_spinbox.value()
            hidden_layers = self.hidden_layers_spinbox.value()
            neurons_per_layer = self.neurons_per_layer_spinbox.value()
            activation_function = ActivationFunction[self.activation_function_combobox.currentText()]
            epochs = self.epochs_spinbox.value()
            learning_rate = self.learning_rate_spinbox.value()

            self.neural_network = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer,
                                                activation_function, mse, mse_prime)
            self.training_thread = TrainingThread(self.neural_network, self.x_data, self.y_data, epochs, learning_rate)
            self.training_thread.started.connect(self.show_training_spinner)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.start()
            self.status_box.setText("Training started")
            self.status_box.setStyleSheet("color: blue")
            self.update_visualization()

    def show_training_spinner(self):
        self.training_dialog = QProgressDialog("Training in progress...", None, 0, 0, self)
        self.training_dialog.setWindowModality(Qt.ApplicationModal)
        self.training_dialog.setCancelButton(None)
        self.training_dialog.setAutoReset(False)
        self.training_dialog.setAutoClose(False)
        self.training_dialog.setWindowTitle("Training")
        self.training_dialog.setFixedSize(300, 100)
        self.training_dialog.show()

    def hide_training_spinner(self):
        if self.training_dialog:
            self.training_dialog.close()

    def on_training_finished(self):
        self.hide_training_spinner()
        self.load_csv_predict_button.setEnabled(True)
        self.status_box.setText("Status: Trained")
        self.status_box.setStyleSheet("color: green")
        self.update_visualization()

    def load_csv_for_prediction(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Prediction CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                data = pd.read_csv(file_path)
                input_size = data.filter(like='Input').shape[1]
                if input_size != self.input_size_spinbox.value():
                    raise ValueError("Prediction CSV input size does not match training input size")

                self.data_loader = CSVDataLoader(file_path, input_size, 0)  # No output size for prediction
                self.data_loader.load_and_check_data()
                self.x_data, _ = self.data_loader.prepare_data()
                if self.neural_network:
                    predictions = self.neural_network.predict(self.x_data)
                    self.format_predictions(predictions)
            except Exception as e:
                self.status_box.setText(f"Error loading CSV for prediction: {e}")

    def format_predictions(self, predictions):
        formatted_predictions = ""
        for idx, prediction in enumerate(predictions):
            formatted_predictions += f"Prediction {idx + 1} -> {prediction[0][0]:.6f}\n"
        self.prediction_output.setText(formatted_predictions)

