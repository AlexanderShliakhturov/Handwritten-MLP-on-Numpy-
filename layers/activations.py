import numpy as np
from MLP.base_classes.activaton import Activation


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)
        super().__init__(relu, relu_prime)
        

# class Softmax(Activation):
#     def __init__(self):
#         softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        
#         def make_jacobian(x):
#             s = softmax(x).reshape(-1, 1)
#             jacobian = np.diagflat(s) - np.dot(s, s.T)
#             return jacobian
                        
#         softmax_prime = make_jacobian

#         super().__init__(softmax, softmax_prime)

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            exp_values = np.exp(x - np.max(x, axis=0, keepdims=True)) 
            return exp_values / np.sum(exp_values, axis=0, keepdims=True)
        
        def softmax_prime(x):
            return 1 
            
        super().__init__(activation=softmax, activation_prime=softmax_prime)
        

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)