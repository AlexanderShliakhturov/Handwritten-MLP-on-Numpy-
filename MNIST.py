import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from layers.dense import Dense
from layers.activations import Tanh, Softmax
from tools.losses import cross_entropy, cross_entropy_prime
from tools.functions import one_hot, predict, get_accuracy, show_and_predict


data = pd.read_csv('Datasets/mnist_train.csv')

X_train = data.drop(columns='label').to_numpy().T
X_train = (X_train/ 255)
Y_mnist = data['label'].to_numpy()
Y_train = np.hstack([one_hot(elem, 10) for elem in Y_mnist])

test_data = pd.read_csv('Datasets/mnist_test.csv')
X_test = test_data.drop(columns='label').to_numpy().T
X_test = (X_test/ 255)
Y_mnist_test = test_data['label'].to_numpy()
Y_test = np.hstack([one_hot(elem, 10) for elem in Y_mnist_test])


network = [Dense(28*28, 256), Tanh(), Dense(256, 128), Tanh(), Dense(128, 10), Softmax()]

epochs = 30
learning_rate = 0.01
batch_size = 64

errors = []
error = 0
for e in tqdm(range(epochs)):
    for i in range(0, X_train.shape[1], batch_size):
#         print(x, y)
        forward_data = X_train[:, i: i + batch_size]
        for layer in network:
            forward_data = layer.forward(forward_data)
        
        error += cross_entropy(Y_train[:, i: i + batch_size], forward_data)

        grad = cross_entropy_prime(Y_train[:, i: i + batch_size], forward_data)

        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= forward_data.shape[1]
    errors.append(error)
    print(f"epoch = {e}, error = {error}")
        
        
plt.plot(range(0, epochs), errors)

get_accuracy(Y_test, predict(network, X_test))

show_and_predict(network, X_test, Y_test, 0)