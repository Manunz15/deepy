import numpy as np
import matplotlib.pyplot as plt

def plot_conf(confusion_matrix: np.ndarray, labels: list) -> None:
    plt.imshow(confusion_matrix, cmap = 'jet')
    plt.colorbar()
    plt.xlabel('Predicted labels', fontweight = 'bold')
    plt.ylabel('True labels', fontweight = 'bold')
    plt.xticks(range(confusion_matrix.shape[0]), labels, rotation = 45 if type(labels[0]) == str else 0)
    plt.yticks(range(confusion_matrix.shape[0]), labels)
    plt.show()