from neural_network.neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, activation_function, loss_function, loss_function_prime):
        """
        Créer un réseau de neurones
        :param input_size: Nombre d'entrées
        :param output_size: Nombre de sorties
        :param hidden_layers: Nombre de couches internes
        :param neurons_per_layer: Nombre de neurones par couche
        :param activation_function: Fonction d'activation choisie
        :param loss_function: Fonction de coût choisie
        :param loss_function_prime: Dérivée de la fonction de coût choisie
        """
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

    def add_layer(self, layer: NeuralLayer) -> None:
        """
        Ajouter une couche au réseau de neurones
        :param layer: Couche de neurones à ajouter
        """
        self.layers.append(layer)

    def predict(self, input_data: list) -> list:
        """
        Réalise une prédiction sur les données d'entrée, en les passant par le réseau de neurones
        :param input_data: Données d'entrée
        :return: Prédictions réalisées par le réseau de neurones
        """
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs: int=20000, learning_rate: float=0.1) -> None:
        """
        Entraîne le réseau de neurones sur les données d'entraînement
        :param x_train: Données d'entraînement
        :param y_train: Sorties attendues
        :param epochs: Nombre d'itérations de l'entraînement
        :param learning_rate: Taux d'apprentissage
        """
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
