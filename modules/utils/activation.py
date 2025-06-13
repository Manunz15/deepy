import numpy as np

class Activation:
    def __init__(self, name: str, func, deriv) -> None:
        self.name: str = name
        self.func: function = func
        self.deriv: function = deriv

class Identity(Activation):
    def __init__(self) -> None:
        def func(z: np.ndarray) -> np.ndarray:
            return z

        def deriv(z: np.ndarray) -> np.ndarray:
            return 1
        
        super().__init__('Identity', func, deriv)

class HiddenSigmoid(Activation):
    # sigmoid for hidden layers
    def __init__(self) -> None:
        def func(z: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-z))

        def deriv(z: np.ndarray) -> np.ndarray:
            return func(z) * func(-z)
        
        super().__init__('Sigmoid', func, deriv)

class Sigmoid(Activation):
    # sigmoid for output of logistic regression
    def __init__(self) -> None:
        def func(z: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-z))

        def deriv(z: np.ndarray) -> int:
            '''
            This is a simplification due to the fact that daJi * dzSigmoid = a - y with
            cross entropy as cost function.
            It does not impact the final result but avoid division by zero.

            See: https://medium.com/data-science/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9
            '''
            return 1
        
        super().__init__('Sigmoid', func, deriv)

class ReLU(Activation):
    def __init__(self) -> None:
        def func(z: np.ndarray) -> np.ndarray:
            return np.maximum(0, z)

        def deriv(z: np.ndarray) -> np.ndarray:
            z_: np.ndarray = np.copy(z)
            z_[z_ > 0] = 1
            z_[z_ < 0] = 0
            return z_
        
        super().__init__('ReLU', func, deriv)

class LReLU(Activation):
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha: float = alpha
        def func(z: np.ndarray) -> np.ndarray:
            return np.maximum(self.alpha * z, z)

        def deriv(z: np.ndarray) -> np.ndarray:
            z_: np.ndarray = np.copy(z)
            z_[z_ > 0] = 1
            z_[z_ <= 0] = self.alpha
            return z_
        
        super().__init__('Leaky ReLU', func, deriv)

class SOFTMAX(Activation):
    def __init__(self, limit: float = 20) -> None:
        self.limit: float = limit
        def func(z: np.ndarray) -> np.ndarray:
            if np.max(z) > self.limit:
                z *= self.limit / np.max(z)
            return np.exp(z) / np.exp(z).sum()

        def deriv(z: np.ndarray) -> int:
            '''
            This is a simplification due to the fact that daJi * dzSOFMAX = a - y with
            cross entropy as cost function.
            It does not impact the final result but reduces computational cost.
            
            See: https://medium.com/@jsilvawasd/softmax-and-backpropagation-625c0c1f8241
            '''
            return 1
        
        super().__init__('SOFTMAX', func, deriv)