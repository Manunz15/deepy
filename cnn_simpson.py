from modules import Network
from modules.fnn import Dense, Flatten
from modules.cnn import Conv3D
from modules.utils import ReLU, LeakyReLU, SOFTMAX, CrossEntropy, plot_conf
from modules.datasets import simpson

# mnist
train_DF, val_DF, test_DF, labels = simpson()

# # network
# net = Network([
#     Conv3D(kernel_shape = (5, 5), 
#                      input_channels = 3, 
#                      output_channels = 8, 
#                      activation = LeakyReLU(),
#                      mode = 'same'),

#     Conv3D(kernel_shape = (7, 7), 
#                     input_channels = 8, 
#                     output_channels = 3, 
#                     activation = LeakyReLU(),
#                     mode = 'same'),
#     Flatten(),
#     Dense(28 * 28 * 3, 500, ReLU()), 
#     Dense(500, 400, ReLU()), 
#     Dense(400, 10, SOFTMAX())],
#     cost_func = CrossEntropy(),
#     save = 'models/cnn_simpson')

# net.fit(train_DF, val_DF, 
#         batch_size = 32, 
#         epochs = 50, 
#         learning_rate = 0.1)

# metrics
net: Network = Network.load('models/cnn_simpson/epoch_9.dpy')
print(f'Test accuracy: {net.calc_accuracy(test_DF):.1f}%')
plot_conf(net.calc_confusion_matrix(test_DF), labels)
net.plot_train()
net.plot_validation('cost')
net.plot_validation('accuracy')