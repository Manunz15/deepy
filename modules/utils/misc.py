import numpy as np

def read_kwargs(kwargs, key, default):
    return default if key not in kwargs else kwargs[key]

def vectorize(y: int | np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if type(y) != np.ndarray and type(y_pred) == np.ndarray:
        y_arr: np.ndarray = np.zeros(len(y_pred))
        y_arr[y] = 1
        return y_arr
    else:
        return y