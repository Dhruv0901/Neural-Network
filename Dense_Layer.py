import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regualriser_l1=0, weight_regulariser_l2=0,
                 bias_regulariser_l1=0, bias_regulariser_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # initial lambda values
        self.weight_regulariser_l1 = weight_regualriser_l1
        self.weight_regulariser_l2 = weight_regulariser_l2
        self.bias_regulariser_l1 = bias_regulariser_l1
        self.bias_regulariser_l2 = bias_regulariser_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # we will update the gradients for weights and biases
        # as the regularisation losses are directly
        # impacting the loss function
        if self.weight_regulariser_l1 > 0: # derivation of sum of all values with respect to a single value
            dL1 = np.ones_like(self.weights) # is 1 for all the positives and -1 for all the negatives
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regulariser_l1 * dL1
        if self.weight_regulariser_l2 > 0: # derivation of lambda * n^2 is 2 * lambda * n
            self.dweights += 2 * self.weight_regulariser_l2 * self.weights
        if self.bias_regulariser_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.weight_regulariser_l1 * dL1
        if self.bias_regulariser_l2 > 0:
            self.dbiases += 2 * self.bias_regulariser_l2 * self.biases


        self.dinputs = np.dot(dvalues, self.weights.T)

