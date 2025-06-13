import numpy as np

class Initialization:
    def __init__(self, name: str, initialize) -> None:
        self.name: str = name
        self.initialize: function = initialize

class Naive(Initialization):
    def __init__(self) -> None:
        def inizialize(n: int, w_shape: tuple[int], b_shape: int) -> tuple:
            w: np.ndarray = np.random.random(w_shape) - 0.5
            b: int | np.ndarray = np.random.random() - 0.5 * (1 if b_shape == 1 else np.ones(b_shape))
            return w, b
        
        super().__init__('Naive', inizialize)

class He(Initialization):
    # He inizialization
    def __init__(self) -> None:
        def inizialize(n: int, w_shape: tuple[int], b_shape: int) -> tuple:
            epsilon: str = np.sqrt(2 / n)
            w: np.ndarray = np.random.normal(0, epsilon, size = w_shape)
            b: int | np.ndarray = 0 if b_shape == 1 else np.zeros(b_shape)
            return w, b
        
        super().__init__('He', inizialize)