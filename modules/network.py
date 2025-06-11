from modules.layer import Layer
from modules.utils.cost import CostFunction
from modules.utils.misc import read_kwargs, vectorize
from modules.utils.metrics import accuracy, confusion_matrix, roc

import numpy as np
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import dill

class Network:
    def __init__(self, layers: list[Layer], cost_func: CostFunction, **kwargs):
        self.layers: list[Layer] = layers
        self.cost_func: CostFunction = cost_func
        self.name: str = read_kwargs(kwargs, 'name', 'Neural Network')
        self.task: str = read_kwargs(kwargs, 'task', 'classification')
        self.folder: str | None = read_kwargs(kwargs, 'save', None)

    def __str__(self) -> str:
        print(f'{"-" * len(self.name)}\n{self.name}\n{"-" * len(self.name)}')
        for layer in self.layers:
            print(layer)

        print(f'\nCost function: {self.cost_func.name}')
        return ''

    def append(self, *args) -> None:
        for layer in args:
            self.layers.append(layer)

    def save_epoch(self, epoch: int) -> None:
        if self.folder is not None:
            filename: str = f'{self.folder}/epoch_{epoch}.dpy'
            self.save(filename)

    def save(self, filename: str) -> None:
        # create folder
        folder: str = os.path.join(*filename.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(__class__, filename: str) -> None:
        with open(filename, 'rb') as f:
            return dill.load(f)

    def calc_accuracy(self, test_DF: np.ndarray) -> float:
        self.accuracy: float = accuracy(self, test_DF)
        return self.accuracy
    
    def calc_confusion_matrix(self, test_DF: np.ndarray) -> np.ndarray:
        classes = 2 if self.task == 'logistic' else len(self.forward(test_DF[0][0]))
        self.confusion_matrix: np.ndarray = confusion_matrix(self, test_DF, classes)
        return self.confusion_matrix
    
    def calc_roc(self, test_DF: np.ndarray, resolution: int = 100) -> tuple[list[int]]:
        self.roc_curve = roc(self, test_DF, resolution)
        return self.roc_curve
    
    def plot_validation(self, metric: str = 'accuracy') -> None:
        title, train, val = {'accuracy': ('Accuracy', self.train_accuracy, self.val_accuracy),
                             'cost':     ('Cost function', self.J_train, self.J_val)}[metric]
        plt.plot(train, label = 'Training', c = "#1679fa")
        plt.plot(val, label = 'Validation', c = "#fa4716")
        plt.xlabel('Epoch', fontweight = 'bold')
        plt.ylabel(title, fontweight = 'bold')
        leg = plt.legend()
        leg.get_frame().set_alpha(0)
        plt.show()

    def plot_train(self) -> None:
        plt.plot(self.every_J, c = '#1679fa')
        plt.xlabel('Batch', fontweight = 'bold')
        plt.ylabel('Training cost function', fontweight = 'bold')
        plt.show()

    def pred(self, X: np.ndarray, threshold: float = 0.5) -> int | np.ndarray:
        self.y_pred: np.ndarray = self.forward(X)

        # classification
        if self.task == 'classification':
            return np.argmax(self.y_pred)
        # logistic
        elif self.task == 'logistic':
            return int(self.y_pred[0] >= threshold)
        # regression
        else:
            return self.y_pred
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)

        y_pred: np.ndarray = X
        return y_pred
    
    def backward(self, y: int | np.ndarray, y_pred: np.ndarray, m: int) -> None:
        y: np.ndarray = vectorize(y, y_pred)

        # first term
        daJi: np.ndarray = self.cost_func.calc_daJi(y, y_pred) / m

        # backprop
        for i in range(len(self.layers)):
            l: int = len(self.layers) - i

            layer: Layer = self.layers[l - 1]
            daJi: np.ndarray = layer.backward(daJi)
    
    def update_weights(self, learning_rate) -> None:
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def fit_batch(self, batch: list[tuple], learning_rate):
        for X, y in batch:
            # forward
            y_pred: np.ndarray = self.forward(X)

            # backward
            self.backward(y, y_pred, len(batch))
        
        self.update_weights(learning_rate)

    def fit(self, train_DF: list[tuple], val_DF: list[tuple], learning_rate = 1, **kwargs):
        epochs: int = read_kwargs(kwargs, 'epochs', 10)
        batch_size: int = read_kwargs(kwargs, 'batch_size', len(train_DF))   # choose between batch and stochastic GD
        batches_number: int = len(train_DF) // batch_size

        # metrics
        self.every_J: list[float] = []
        self.J_train: list[float] = []
        self.J_val: list[float] = []
        self.train_accuracy: list[float] = []
        self.val_accuracy:   list[float] = []

        # train the network
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}')
            shuffle(train_DF)

            for i in tqdm(range(batches_number)):
                batch = train_DF[i * batch_size: (i + 1) * batch_size]
                self.every_J.append(self.cost_func.calc_J(self, batch))
                self.fit_batch(batch, learning_rate)
                
            # metrics
            self.J_train.append(self.cost_func.calc_J(self, train_DF))
            self.J_val.append(self.cost_func.calc_J(self, val_DF))
            train_accuracy: float = accuracy(self, train_DF)
            val_accuracy:   float = accuracy(self, val_DF)
            self.train_accuracy.append(train_accuracy)
            self.val_accuracy.append(val_accuracy)
            print(f'Train: {train_accuracy:.1f}% | Validation: {val_accuracy:.1f}%')

            # early stopping
            self.save_epoch(epoch)

    