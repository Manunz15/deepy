from modules.utils.activation import Activation
from modules.utils.initialization import Initialization
from modules.layer import Layer
from modules.utils.misc import read_kwargs

import numpy as np
from scipy.signal import correlate, convolve

class Conv3D(Layer):
    def __init__(self, kernel_shape: tuple[int], input_channels: int, output_channels: int, activation: Activation, **kwargs):
        super().__init__(activation, **kwargs)

        # initialization
        self.kernel_shape: tuple[int] = kernel_shape
        self.input_channels: int = input_channels
        self.output_channels: int = output_channels  # number of filters
        
        # read kwargs
        self.mode: str = read_kwargs(self.kwargs, 'mode', 'valid')
        self.pad:  int = read_kwargs(self.kwargs, 'pad',  0)
        self.padx: int = read_kwargs(self.kwargs, 'padx', max(self.pad, 0))
        self.pady: int = read_kwargs(self.kwargs, 'pady', max(self.pad, 0))
        self.zero_is_mean: bool = read_kwargs(self.kwargs, 'zero_is_mean',  False)

        self.create_kernels()

    def __str__(self):
        return f'Convolutional: {self.output_channels} filters {self.kernel_shape[0]}x{self.kernel_shape[1]}x{self.input_channels} | {self.activation.name}'

    def create_kernels(self) -> None:
        self.kernels: list[Kernel] = []
        for _ in range(self.output_channels):
            self.kernels.append(Kernel(self.kernel_shape, self.input_channels, self.init))

    def padding(self, X: np.ndarray) -> np.ndarray:
        # keep shape
        if self.mode == 'same':
            self.padx = (self.kernel_shape[0] - 1) // 2
            self.pady = (self.kernel_shape[1] - 1) // 2

        # explicit padding
        if self.padx or self.pady:   
            if self.zero_is_mean:
                X = X - np.mean(X) 
            X = np.pad(X, ((self.padx, self.padx), (self.pady, self.pady), (0, 0)))

        return X
    
    def unpadding(self, daJi: np.ndarray) -> np.ndarray:
        if self.padx or self.pady:   
            daJi: np.ndarray = daJi[self.padx:-self.padx, self.pady:-self.pady, :]
        return daJi

    def forward(self, X: np.ndarray) -> np.ndarray:
        # reshape X and pad
        X = self.padding(X.reshape(X.shape[0], X.shape[1], self.input_channels))
        super().forward(X)

        # create feature maps
        feature_maps: list[np.ndarray] = []
        for kernel in self.kernels:
            feature_maps.append(kernel.forward(self.X))

        # concatenate output channels
        self.z: np.ndarray = np.concatenate(feature_maps, axis = 2)
        self.a: np.ndarray = self.activation.func(self.z)
        
        return self.a
    
    def backward(self, daJi: np.ndarray) -> np.ndarray:
        # error
        delta: np.ndarray = (daJi * self.activation.deriv(self.z)).reshape(daJi.shape[0], daJi.shape[1], daJi.shape[2], 1)

        # gradients of kernels and daJi
        daJi: np.ndarray = 0
        for i, kernel in enumerate(self.kernels):
            kernel.backward(self.X, delta[:,:,i])
            daJi += np.concatenate([kernel.convolve(delta[:,:,i], kernel.w[:,:,j].reshape(self.kernel_shape[0], self.kernel_shape[0], 1)) 
                                    for j in range(kernel.w.shape[2])], axis = 2)

        daJi: np.ndarray = self.unpadding(daJi)
        return daJi
    
    def update_weights(self, learning_rate: float) -> None:
        for kernel in self.kernels:
            kernel.update_weights(learning_rate)

class Kernel:
    def __init__(self, kernel_shape: tuple[int], input_channels: int, init: Initialization):
        # initialization   
        self.kernel_shape: tuple[int] = kernel_shape
        self.input_channels: int = input_channels

        w_shape = (*self.kernel_shape, self.input_channels)
        init: tuple[np.ndarray] = init.initialize(np.prod(w_shape), w_shape, 1)
        self.w: np.ndarray = init[0]
        self.b: np.ndarray = init[1]

        # cost function
        self.dwJi: np.ndarray = 0
        self.dbJi: float = 0

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
        return self.correlate(X, self.w) + self.b
    
    def backward(self, X: np.ndarray, delta: np.ndarray) -> None:    
        self.dwJi += self.correlate(X, delta)
        self.dbJi += np.sum(delta)

    def update_weights(self, learning_rate: float) -> None:
        # update weights and biases
        self.w: np.ndarray = self.w - learning_rate * self.dwJi
        self.b: np.ndarray = self.b - learning_rate * self.dbJi

        # reset gradients
        self.dwJi: np.ndarray = 0
        self.dbJi: np.ndarray = 0