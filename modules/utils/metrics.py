from modules.utils.misc import read_kwargs

from typing import Callable
import numpy as np

def accuracy(network: Callable, test_DF: list[tuple]) -> float:
    accuracy: float = 0
    for X, y in test_DF:
        y_pred: int = network.pred(X)
        accuracy += 100 / len(test_DF) if y == y_pred else 0

    return accuracy

def confusion_matrix(network: Callable, test_DF: list[tuple], classes: int, **kwargs) -> float:
    threshold: float = read_kwargs(kwargs, 'threshold', 0.5)
    confusion_matrix: np.ndarray = np.zeros([classes, classes])
    for X, y in test_DF:
        y_pred: int = network.pred(X, threshold)
        confusion_matrix[y, y_pred] += 1

    confusion_matrix /= np.sum(confusion_matrix, axis = 1)

    return confusion_matrix

def roc(network: Callable, test_DF: list[tuple], resolution: int = 100) -> tuple[list[float]]:
    FP_rate: list[float] = []
    TP_rate: list[float] = []
    for threshold in np.linspace(0, 1, resolution):
        conf_matrix: np.ndarray = confusion_matrix(network, test_DF, 2, threshold = threshold)
        FP_rate.append(conf_matrix[0, 1])
        TP_rate.append(conf_matrix[1, 1])

    return FP_rate, TP_rate