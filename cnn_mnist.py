from modules import Network
from modules.fnn import Dense, Flatten
from modules.cnn import Conv3D
from modules.utils import ReLU, LeakyReLU, SOFTMAX, CrossEntropy, plot_conf
from modules.datasets.mnist.mnist import mnist

# mnist
train_DF, val_DF, test_DF, labels = mnist()

# # network
# net = Network([
#     Conv3D(kernel_shape = (5, 5), 
#                      input_channels = 1, 
#                      output_channels = 3, 
#                      activation = LeakyReLU()),

#     Conv3D(kernel_shape = (7, 7), 
#                     input_channels = 3, 
#                     output_channels = 2, 
#                     activation = LeakyReLU()),
#     Flatten(),
#     Dense(18 * 18 * 2, 100, ReLU()), 
#     Dense(100, 10, SOFTMAX())],
#     cost_func = CrossEntropy(),
#     save = 'models/cnn_mnist')

# net.fit(train_DF, val_DF, 
#         batch_size = 32, 
#         epochs = 25, 
#         learning_rate = 0.3)

# full trained
net: Network = Network.load('models/cnn_mnist/epoch_24.dpy')
net.plot_train()
net.plot_validation('cost')
net.plot_validation('accuracy')

# optimal
net: Network = Network.load('models/cnn_mnist/epoch_1.dpy')
print(f'Test accuracy: {net.calc_accuracy(test_DF):.1f}%')
plot_conf(net.calc_confusion_matrix(test_DF), labels)