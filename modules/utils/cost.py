import numpy as np
from typing import Callable

from modules.utils.misc import vectorize

class CostFunction:
    def __init__(self, name, calc_Ji, calc_daJi):
        self.name: str = name
        self.calc_Ji: function = calc_Ji
        self.calc_daJi: function = calc_daJi

    def calc_J(self, network: Callable, DF: list[tuple]) -> float:
        J: float = 0
        for X, y in DF:
            # vectorize y
            y_pred: np.ndarray = network.forward(X)
            y = vectorize(y, y_pred)

            J += self.calc_Ji(y, y_pred) / len(DF)

        return J

class MSE(CostFunction):
    def __init__(self):
        def calc_Ji(y: np.ndarray, a: np.ndarray) -> float:
            return 0.5 * np.linalg.norm(y - a) ** 2
        
        def calc_daJi(y: np.ndarray, a: np.ndarray) -> np.ndarray:
            return a - y
        
        super().__init__('MSE', calc_Ji, calc_daJi)

class CrossEntropy(CostFunction):
    def __init__(self):
        def calc_Ji(y: np.ndarray, a: np.ndarray) -> float:
            return - np.sum(y * np.log(a))
        
        def calc_daJi(y: np.ndarray, a: np.ndarray) -> np.ndarray:
            '''
            This is a simplification due to the fact that daJi * dzSOFMAX = a - y.
            It does not impact the final result but reduces computational cost.

            See: https://medium.com/@jsilvawasd/softmax-and-backpropagation-625c0c1f8241
            '''
            return a - y
        
        super().__init__('Cross entropy', calc_Ji, calc_daJi)