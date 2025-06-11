from modules.utils.activation import Activation
from modules.layer import Layer

import numpy as np

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, activation: Activation, **kwargs) -> None:
        super().__init__(activation, **kwargs)

        # initialization
        self.input_size: int = input_size
        self.output_size: int = output_size

        w_shape: tuple[int] = (self.output_size, self.input_size)
        init: tuple[np.ndarray] = self.init.initialize(np.prod(w_shape), w_shape, self.output_size)
        self.w: np.ndarray = init[0]
        self.b: np.ndarray = init[1]

        # cost function
        self.dwJi: np.ndarray = 0
        self.dbJi: np.ndarray = 0

    def __str__(self) -> str:
        return f'Dense: {self.output_size} perceptrons | {self.activation.name}'

    def forward(self, X: np.ndarray) -> np.ndarray: 
        super().forward(X)
        self.z: np.ndarray = np.matmul(self.w, self.X) + self.b
        self.a: np.ndarray = self.activation.func(self.z)

        return self.a
    
    def backward(self, daJi: np.ndarray) -> np.ndarray:
        # error
        delta: np.ndarray = daJi * self.activation.deriv(self.z)

        # gradients
        self.dwJi += np.matmul(delta.reshape(-1, 1), self.X.reshape(1, -1))
        self.dbJi += delta

        # dJ/da for the next (previous) layer
        daJi: np.ndarray = np.matmul(delta, self.w)

        return daJi
    
    def update_weights(self, learning_rate: float) -> None:
        # update weights and biases
        self.w: np.ndarray = self.w - learning_rate * self.dwJi
        self.b: np.ndarray = self.b - learning_rate * self.dbJi

        # reset gradients
        self.dwJi: np.ndarray = 0
        self.dbJi: np.ndarray = 0