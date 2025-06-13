from modules.utils.activation import Activation
from modules.utils.initialization import Initialization, He, Naive
from modules.utils.misc import read_kwargs

import numpy as np

class Layer:
    def __init__(self, activation: Activation, **kwargs) -> None:
        # initialization
        self.activation: Activation | None = activation
        self.kwargs = kwargs

        init_name: str = read_kwargs(kwargs, 'init', 'he')
        self.init: Initialization = {'naive': Naive(),
                                     'he': He()
                                     }[init_name]

    def forward(self, X: np.ndarray) -> None:
        self.X: np.ndarray = X

    def backward(self, daJi: np.ndarray) -> np.ndarray:
        return daJi
    
    def update_weights(self, *args, **kwargs) -> None:
        pass