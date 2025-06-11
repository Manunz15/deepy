from modules.layer import Layer
from modules.utils.misc import read_kwargs

import numpy as np
from skimage.measure import block_reduce

class Pooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)

        # initialization
        self.block_shape: tuple[int] | None = read_kwargs(self.kwargs, 'block_shape', None)
        self.final_shape: tuple[int] | None = read_kwargs(self.kwargs, 'final_shape', None)
        self.method: str = read_kwargs(self.kwargs, 'method', 'max')
        self.methods_func: dict[str, function] = {'max': np.max, 'average': np.mean}[self.method]

    def __str__(self):
        if self.final_shape is not None:
            return f'Pooling: final {self.final_shape[0]}x{self.final_shape[1]}'
        else:
            return f'Pooling: final {self.block_shape[0]}x{self.block_shape[1]}'
    
    def setup(self, X: np.ndarray) -> np.ndarray:
        if self.final_shape is not None:
            # if it can be divided
            if not X.shape[0] % self.final_shape[0] and not X.shape[1] % self.final_shape[1]:
                self.block_shape = (X.shape[0] // self.final_shape[0], X.shape[1] // self.final_shape[1])
            # if not -> padding
            else:
                self.block_shape = (X.shape[0] // self.final_shape[0] + 1, X.shape[1] // self.final_shape[1] + 1)
                padx = self.final_shape[0] - X.shape[0] % self.final_shape[0]
                pady = self.final_shape[1] - X.shape[1] % self.final_shape[1]
                X = np.pad(X, ((padx // 2, padx - padx // 2), (pady // 2, pady - pady // 2), (0, 0)))

        return X

    def forward(self, X: np.ndarray) -> np.ndarray:
        # setup block shape and X
        X = self.setup(X)
        super().forward(X)

        # pooling
        self.a: np.ndarray = block_reduce(self.X, (*self.block_shape, 1), self.methods_func)

        return self.a
    
    def backward(self, daJi: np.ndarray) -> np.ndarray:
        print(self.X.shape, daJi.shape)
        return 0