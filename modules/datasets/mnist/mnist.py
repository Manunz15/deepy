import numpy as np
import pandas as pd

def mnist() -> list[tuple]:
    MNIST: pd.DataFrame = pd.read_csv('modules/datasets/mnist/mnist.csv')
    data_list: list[tuple] = []

    for row in MNIST.iterrows():
        image: np.ndarray = row[1].to_numpy()
        data_list.append((image[1:].reshape(28, 28) / 255, image[0]))

    labels: list[int] = [i for i in range(10)]

    return data_list[:35200], data_list[35200:38000], data_list[38000:], labels