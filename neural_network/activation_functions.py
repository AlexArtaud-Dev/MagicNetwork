from enum import Enum
import numpy as np

# ReLU (Rectified Linear Unit) :
#     Sortie : [0,+∞)[0,+∞)
#     Dérivée : 11 si x>0x>0, sinon 00
#     Utilisée pour : Capturer les non-linéarités, éviter le problème de gradient vanishing
#
# Tanh (Tangente hyperbolique) :
#     Sortie : [−1,1][−1,1]
#     Dérivée : 1−tanh⁡(x)21−tanh(x)2
#     Utilisée pour : Normaliser les entrées entre [−1,1][−1,1], souvent dans les couches cachées
#
# Sigmoïde :
#     Sortie : (0,1)(0,1)
#     Dérivée : σ(x)⋅(1−σ(x))σ(x)⋅(1−σ(x)) où σ(x)σ(x) est la fonction sigmoïde
#     Utilisée pour : Normaliser les sorties, souvent pour des problèmes de classification binaire
#
# Leaky ReLU :
#     Sortie : (−∞,+∞)(−∞,+∞) (mais principalement [0,+∞)[0,+∞) avec une petite pente négative)
#     Dérivée : 11 si x>0x>0, sinon αα (souvent α=0.01α=0.01)
#     Utilisée pour : Éviter les morts de neurones (problème de gradient mort)

class ActivationFunction(Enum):
    RELU = 1
    TANH = 2
    SIGMOID = 3
    LEAKY_RELU = 4

def get_activation_function(activation_function):
    if activation_function == ActivationFunction.RELU:
        return relu, relu_prime
    elif activation_function == ActivationFunction.TANH:
        return tanh, tanh_prime
    elif activation_function == ActivationFunction.SIGMOID:
        return sigmoid, sigmoid_prime
    elif activation_function == ActivationFunction.LEAKY_RELU:
        return leaky_relu, leaky_relu_prime
    else:
        raise ValueError("Unsupported activation function")

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
