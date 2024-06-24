
# 🌟 Projet de Réseau de Neurones 🌟

## 📝 Vue d'ensemble

Ce projet est une implémentation d'un réseau de neurones à partir de zéro en utilisant Python. L'implémentation inclut des couches, des neurones, des fonctions d'activation et des fonctions de perte personnalisées. Le projet comprend également une interface graphique (GUI) pour configurer et entraîner le réseau de neurones, ainsi qu'une interface en ligne de commande (CLI) pour l'exécution en ligne de commande.

## ✨ Fonctionnalités

- 🚀 **Multiples fonctions d'activation** : ReLU, Tanh, Sigmoid, Leaky ReLU
- 📉 **Fonction de perte** : Erreur Quadratique Moyenne (MSE)
- 🔗 **Couches entièrement connectées**
- 🛠️ **Facile à utiliser et à étendre**


## 📦 Installation

Pour exécuter le projet, vous devez avoir Python installé. Vous pouvez installer les dépendances requises en utilisant la commande suivante :

```bash
pip install -r requirements.txt
```

## 🗂️ Structure du Projet

Le projet est organisé en plusieurs modules Python :

- `activation_functions.py` : Contient diverses fonctions d'activation et leurs dérivées.
- `csv_dataloader.py` : Gère le chargement et la validation des données à partir de fichiers CSV.
- `loss_functions.py` : Contient des fonctions de perte et leurs dérivées.
- `neural_layer.py` : Définit les couches du réseau de neurones.
- `neural_network.py` : Implémente le réseau de neurones.
- `neuron.py` : Définit les neurones individuels.
- `main_window.py` : Implémentation de l'interface graphique pour configurer et entraîner le réseau de neurones.
- `training_thread.py` : Gère le processus d'entraînement dans un thread séparé pour l'interface graphique.
- `runner.py` : Contient différents modes d'exécution (CLI, GUI, TEST).
- `main.py` : Point d'entrée pour exécuter le projet.

## ⚙️ Configuration du Réseau de Neurones

### Paramètres

- `input_size` : Nombre de caractéristiques d'entrée.
- `output_size` : Nombre de caractéristiques de sortie.
- `hidden_layers` : Nombre de couches cachées dans le réseau.
- `neurons_per_layer` : Nombre de neurones par couche cachée.
- `activation_function` : Fonction d'activation à utiliser (RELU, TANH, SIGMOID, LEAKY_RELU).
- `loss_function` : Fonction de perte à utiliser (actuellement, seule MSE est implémentée).
- `epochs` : Nombre d'itérations d'entraînement.
- `learning_rate` : Taux d'apprentissage.

### ⚡ Fonctions d'Activation

Le framework supporte les fonctions d'activation suivantes :

1. **ReLU (Rectified Linear Unit)**
    - **Sortie** : [0, +∞)
    - **Dérivée** : 1 si x > 0, sinon 0
    - **Cas d'utilisation** : Capturer les non-linéarités, éviter le problème de gradient vanishing

2. **Tanh (Tangente Hyperbolique)**
    - **Sortie** : [-1, 1]
    - **Dérivée** : 1 - tanh(x)^2
    - **Cas d'utilisation** : Normaliser les entrées entre [-1, 1], souvent utilisé dans les couches cachées

3. **Sigmoïde**
    - **Sortie** : (0, 1)
    - **Dérivée** : σ(x) * (1 - σ(x)), où σ(x) est la fonction sigmoïde
    - **Cas d'utilisation** : Normaliser les sorties, souvent utilisé pour des problèmes de classification binaire

4. **Leaky ReLU**
    - **Sortie** : (-∞, +∞) (mais principalement [0, +∞) avec une petite pente négative)
    - **Dérivée** : 1 si x > 0, sinon α (souvent α = 0.01)
    - **Cas d'utilisation** : Éviter les neurones morts (problème de gradient mort)

### 🎯 Obtenir de Bons Résultats

Pour obtenir de bons résultats pour différents types de données, considérez les conseils suivants :

1. **📊 Normalisation des Données** : Assurez-vous que vos données d'entrée sont normalisées. Par exemple, échellez vos caractéristiques pour avoir une moyenne nulle et une variance unitaire.
2. **⚙️ Fonction d'Activation** : Choisissez la fonction d'activation qui convient le mieux à votre problème. Par exemple, utilisez Sigmoid ou Tanh pour les tâches de classification binaire et ReLU pour les réseaux profonds.
3. **🏗️ Architecture du Réseau** : Expérimentez avec le nombre de couches et de neurones par couche. Les tâches plus complexes peuvent nécessiter des réseaux plus profonds.
4. **🧠 Taux d'Apprentissage** : Ajustez le taux d'apprentissage. Un taux trop élevé peut entraîner une convergence trop rapide vers une solution sous-optimale, tandis qu'un taux trop bas peut rendre le processus d'entraînement trop lent.
5. **⏳ Époques** : Entraînez pendant un nombre suffisant d'époques. Cependant, méfiez-vous du surapprentissage ; envisagez d'utiliser l'arrêt précoce ou la validation croisée pour trouver le nombre optimal d'époques.


## 🚀 Entraînement et Test

### Format des Données d'Entraînement

Les données d'entraînement doivent être au format CSV avec les caractéristiques d'entrée et les valeurs de sortie. Les caractéristiques d'entrée doivent être préfixées par 'Input' et les caractéristiques de sortie par 'Output'.

Exemple de fichier CSV d'entraînement (`train_data_relu.csv`) :

```csv
Input1,Input2,Output
0,0,0
0,1,1
1,0,1
1,1,0
```

### Format des Données de Test

Les données de test doivent suivre le même format que les données d'entraînement, mais sans les valeurs de sortie.

Exemple de fichier CSV de test (`test_data_relu.csv`) :

```csv
Input1,Input2
0,0
0,1
1,0
1,1
```

## 💻 Utilisation

### Interface Graphique (GUI)

Pour lancer l'application avec l'interface graphique, exécutez le script `main.py` :

```bash
python main.py
```

L'application GUI permet de configurer les paramètres du réseau de neurones, de charger des fichiers CSV pour l'entraînement et les prédictions, et de visualiser la structure du réseau.

### Ligne de Commande (CLI)

Pour exécuter l'application en ligne de commande, modifiez le script `main.py` pour utiliser le mode CLI et exécutez-le :

```bash
python main.py
```

Exemple d'exécution en ligne de commande avec les données XOR :

```python
def cli():
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
    hidden_layers = 1
    neurons_per_layer = 2
    activation_function = ActivationFunction.LEAKY_RELU

    net = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer, activation_function, mse, mse_prime)
    net.fit(x_train, y_train, epochs=100000, learning_rate=0.1)

    out = net.predict(x_train)
    print(out)
```

Fichier `requirements.txt` :

```text
pandas~=2.2.2
PySide6~=6.7.2
numpy~=2.0.0
```


Des fichiers de test sont inclus dans le dossier `tests/datasets` du projet, et les configurations + résultats des tests sont stockés dans le dossier `tests/results`.
