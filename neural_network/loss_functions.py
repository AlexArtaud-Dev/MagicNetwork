import numpy as np

def mse(y_true: float, y_pred: float) -> float:
    """
    Fonction de coût Mean Squared Error
    :param y_true: Valeur réelle
    :param y_pred: Valeur prédite
    :return: La moyenne des erreurs au carré
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true: float, y_pred: float) -> float:
    """
    Dérivée de la fonction de coût Mean Squared Error
    :param y_true: Valeur réelle
    :param y_pred: Valeur prédite
    :return: La moyenne des erreurs passée par la dérivée de la fonction MSE
    """
    return 2 * (y_pred - y_true) / y_true.size
