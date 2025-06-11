import os
import cv2
from random import shuffle, seed

def simpson(color: str = 'rgb') -> list[tuple]:
    path: str = os.path.join('modules/datasets/simpson/', color)
    test_data: list[tuple] = []
    train_data: list[tuple] = []
    
    print('Loading the Simpson MNIST dataset...')
    for name, dataset in zip(['test', 'train'], [test_data, train_data]):
        for i, character in enumerate(os.listdir(os.path.join(path, name))):
            for image in os.listdir(os.path.join(path, name, character)):
                dataset.append((cv2.imread(os.path.join(path, name, character, image)) / 255, i))
    print('Done!')

    # to have a validation set
    seed(42)
    shuffle(train_data)

    labels: list[str] = ['Bart', 'Montgomery', 'Homer', 'Krusty', 'Lisa', 'Marge', 'Milhouse', 'Moe', 'Flanders', 'Skinner']

    return train_data[:7200], train_data[7200:], test_data, labels