import numpy as np
from base_classes.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
#         self.weights = np.random.randn(output_size, input_size)
#         self.bias = np.random.randn(output_size, 1)
#         self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
#         self.bias = np.zeros((output_size, 1))
        self.weights = np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        dEdX = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient        
#         self.bias -= learning_rate * output_gradient
        self.bias -= learning_rate * np.reshape(np.sum(output_gradient, axis= 1), (output_gradient.shape[0], 1))

        return dEdX