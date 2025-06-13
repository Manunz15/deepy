from modules import Network
from modules.fnn import Dense, Flatten
from modules.utils import ReLU, SOFTMAX, CrossEntropy, plot_conf
from modules.datasets import simpson

# mnist
train_DF, val_DF, test_DF, labels = simpson()

# # network
# net = Network([
#     Flatten(),
#     Dense(28 * 28 * 3, 500, ReLU()),
#     Dense(500, 400, ReLU()),
#     Dense(400, 10, SOFTMAX())],
#     cost_func = CrossEntropy(),
#     save = 'models/fnn_simpson')

# net.fit(train_DF, val_DF, 
#         batch_size = 32, 
#         epochs = 50, 
#         learning_rate = 0.1)

# metrics
net: Network = Network.load('models/fnn_simpson/epoch_23.dpy')
print(f'Test accuracy: {net.calc_accuracy(test_DF):.1f}%')
plot_conf(net.calc_confusion_matrix(test_DF), labels)
net.plot_train()
net.plot_validation('cost')
net.plot_validation('accuracy')