import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power((y_true - y_pred), 2))
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true)/np.size(y_true)


def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
def cross_entropy_prime(y_true, y_pred):
#     return -y_true / y_pred
    return y_pred - y_true