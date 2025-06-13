from modules import Network
from modules.fnn import Dense, Flatten
from modules.utils.activation import ReLU, SOFTMAX
from modules.utils.cost import CrossEntropy
from modules.utils.plot import plot_conf
from modules.datasets import mnist

# mnist
train_DF, val_DF, test_DF, labels = mnist()

# # network
# net = Network([
#     Flatten(),
#     Dense(28 * 28, 100, ReLU()),
#     Dense(100, 20, ReLU()),
#     Dense(20, 10, SOFTMAX())],
#     cost_func = CrossEntropy(),
#     save = 'models/fnn_mnist')

# net.fit(train_DF, val_DF, 
#         batch_size = 32, 
#         epochs = 25, 
#         learning_rate = 0.4)

# metrics
net: Network = Network.load('models/fnn_mnist/epoch_3.dpy')
print(f'Test accuracy: {net.calc_accuracy(test_DF):.1f}%')
plot_conf(net.calc_confusion_matrix(test_DF), labels)
net.plot_train()
net.plot_validation('cost')
net.plot_validation('accuracy')