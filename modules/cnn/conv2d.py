from modules.utils.activation import Activation
from modules.layer import Layer
from modules.utils.misc import read_kwargs

import numpy as np
from scipy.signal import correlate, convolve

class Conv2D(Layer):
    def __init__(self, kernel_shape: tuple[int], activation: Activation, **kwargs):
        super().__init__(activation, **kwargs)

        # initialization
        self.kernel_shape: tuple[int] = kernel_shape
        init: tuple[np.ndarray] = self.init.initialize(np.prod(self.kernel_shape), self.kernel_shape, 1)
        self.w: np.ndarray = init[0]
        self.b: np.ndarray = init[1]

        # cost function
        self.dwJi: np.ndarray = 0
        self.dbJi: float = 0

        # read kwargs
        self.mode: str = read_kwargs(self.kwargs, 'mode', 'valid')
        self.pad:  int = read_kwargs(self.kwargs, 'pad',  0)
        self.padx: int = read_kwargs(self.kwargs, 'padx', max(self.pad, 0))
        self.pady: int = read_kwargs(self.kwargs, 'pady', max(self.pad, 0))
        self.zero_is_mean: bool = read_kwargs(self.kwargs, 'zero_is_mean',  False)

    def __str__(self):
        return f'Convolutional 2D: {self.kernel_shape[0]}x{self.kernel_shape[1]} | {self.activation.name}'

    def padding(self, X: np.ndarray) -> np.ndarray:
        # keep shape
        if self.mode == 'same':
            self.padx = (self.kernel_shape[0] - 1) // 2
            self.pady = (self.kernel_shape[1] - 1) // 2

        # explicit padding
        if self.padx or self.pady:   
            if self.zero_is_mean:
                X = X - np.mean(X) 
            X = np.pad(X, ((self.padx, self.padx), (self.pady, self.pady)))

        return X
    
    def unpadding(self, daJi: np.ndarray) -> np.ndarray:
        if self.padx or self.pady:   
            daJi: np.ndarray = daJi[self.padx:-self.padx, self.pady:-self.pady]
        return daJi
    
    def convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''
        "full" is needed to return to the original X shape (included padding)
        '''
        return convolve(a, b, mode = 'full')
    
    def correlate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''
        "valid" is needed to make to return a 2D feature map, even with 3D inputs
        '''
        return correlate(a, b, mode = 'valid')

    def forward(self, X: np.ndarray) -> np.ndarray:
        # add padding
        super().forward(self.padding(X))

        # forward
        self.z: np.ndarray = self.correlate(X, self.w) + self.b
        self.a: np.ndarray = self.activation.func(self.z)
        
        return self.a
    
    def backward(self, daJi: np.ndarray) -> np.ndarray:
        # error
        delta: np.ndarray = daJi * self.activation.deriv(self.z)

        # gradients of kernel
        self.dwJi += self.correlate(self.X, delta)
        self.dbJi += np.sum(delta)

        # daJi
        daJi: np.ndarray = self.unpadding(self.convolve(delta, self.w))
        return daJi
    
    def update_weights(self, learning_rate: float) -> None:
        # update weights and biases
        self.w: np.ndarray = self.w - learning_rate * self.dwJi
        self.b: np.ndarray = self.b - learning_rate * self.dbJi

        # reset gradients
        self.dwJi: np.ndarray = 0
        self.dbJi: np.ndarray = 0