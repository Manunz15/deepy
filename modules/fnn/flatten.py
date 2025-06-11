from modules.layer import Layer

import numpy as np

class Flatten(Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(None, **kwargs)

    def __str__(self) -> str:
        return f'Flatten'

    def forward(self, X: np.ndarray) -> np.ndarray:
        super().forward(X)
        self.a: np.ndarray = self.X.reshape(-1)

        return self.a
    
    def backward(self, daJi: np.ndarray) -> np.ndarray:
        return daJi.reshape(self.X.shape)