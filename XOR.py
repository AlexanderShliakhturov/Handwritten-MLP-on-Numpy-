import numpy as np
import matplotlib.pyplot as plt

from MLP.layers.dense import Dense
from MLP.layers.activations import Tanh
from MLP.tools.losses import mse, mse_prime

X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
Y = np.reshape([[0], [1], [1], [0]], (4,1,1))

network_xor = [Dense(2, 4), Tanh(), Dense(4,1), Tanh()]


epochs = 100
learning_rate = 0.1
errors = []
error = 0

for e in range(epochs):
    for x, y in zip(X, Y):
        forward_data = x
        for layer in network_xor:
            forward_data = layer.forward(forward_data)
        
        error += mse(y, forward_data)

        grad = mse_prime(y, forward_data)

        for layer in reversed(network_xor):
            grad = layer.backward(grad, learning_rate)
            

    error /= len(X)
    errors.append(error)
    print(f"epoch = {e}, error = {error}")
        
        
plt.plot(range(0, epochs), errors)