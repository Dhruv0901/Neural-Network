import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regualriser_l1=0, weight_regulariser_l2=0,
                 bias_regulariser_l1=0, bias_regulariser_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # initial lambda values
        self.weight_regulariser_l1 = weight_regualriser_l1
        self.weight_regulariser_l2 = weight_regulariser_l2
        self.bias_regulariser_l1 = bias_regulariser_l1
        self.bias_regulariser_l2 = bias_regulariser_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
