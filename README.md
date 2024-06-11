
# 🌟 Framework de Réseau de Neurones 🌟

Bienvenue dans ce framework de réseau de neurones simple et efficace. Que vous soyez un débutant ou un expert en apprentissage automatique, ce framework vous permet de créer, entraîner et évaluer des réseaux de neurones avec facilité.

## ✨ Fonctionnalités

- 🚀 **Multiples fonctions d'activation** : ReLU, Tanh, Sigmoid, Leaky ReLU
- 📉 **Fonction de perte** : Erreur Quadratique Moyenne (MSE)
- 🔗 **Couches entièrement connectées**
- 🛠️ **Facile à utiliser et à étendre**

## 🚀 Installation

1. Clonez le dépôt.
2. Assurez-vous d'avoir les packages requis installés :
    ```bash
    pip install numpy
    ```

## 📚 Utilisation

### Exemple

Voici un exemple d'utilisation du framework avec un problème simple de XOR :

```python
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
    hidden_layers = 1
    neurons_per_layer = 2
    activation_function = ActivationFunction.LEAKY_RELU

    net = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer, activation_function, mse, mse_prime)
    net.fit(x_train, y_train, epochs=10000, learning_rate=0.01)

    out = net.predict(x_train)
    print(out)
```

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

### 🔧 Paramètres

- **input_size** : Nombre de caractéristiques d'entrée
- **output_size** : Nombre de caractéristiques de sortie
- **hidden_layers** : Nombre de couches cachées
- **neurons_per_layer** : Nombre de neurones par couche cachée
- **activation_function** : Fonction d'activation à utiliser (ReLU, Tanh, Sigmoid, Leaky ReLU)
- **loss_function** : Fonction de perte à utiliser (actuellement seule la MSE est implémentée)
- **loss_function_prime** : Dérivée de la fonction de perte
- **epochs** : Nombre d'itérations d'entraînement
- **learning_rate** : Taux d'apprentissage pour la mise à jour des poids

### 🎯 Obtenir de Bons Résultats

Pour obtenir de bons résultats pour différents types de données, considérez les conseils suivants :

1. **📊 Normalisation des Données** : Assurez-vous que vos données d'entrée sont normalisées. Par exemple, échellez vos caractéristiques pour avoir une moyenne nulle et une variance unitaire.
2. **⚙️ Fonction d'Activation** : Choisissez la fonction d'activation qui convient le mieux à votre problème. Par exemple, utilisez Sigmoid ou Tanh pour les tâches de classification binaire et ReLU pour les réseaux profonds.
3. **🏗️ Architecture du Réseau** : Expérimentez avec le nombre de couches et de neurones par couche. Les tâches plus complexes peuvent nécessiter des réseaux plus profonds.
4. **🧠 Taux d'Apprentissage** : Ajustez le taux d'apprentissage. Un taux trop élevé peut entraîner une convergence trop rapide vers une solution sous-optimale, tandis qu'un taux trop bas peut rendre le processus d'entraînement trop lent.
5. **⏳ Époques** : Entraînez pendant un nombre suffisant d'époques. Cependant, méfiez-vous du surapprentissage ; envisagez d'utiliser l'arrêt précoce ou la validation croisée pour trouver le nombre optimal d'époques.

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🤝 Contribuer

Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request pour toute amélioration ou correction de bug.

