import numpy as np


class Optimiser_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):# step learning decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # cache is like a record that keeps up with the changes
        # so if the changes are massive (either +ve or -ve) further updates would be minimised
        if not hasattr(layer, 'weight_cache'):# initialises an array if it does not exist before
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = layer.dweights ** 2# layer's weight cache is the square of weight gradients
        layer.bias_cache = layer.dbiases ** 2# similarly layer's bias cache is the square of bias gradients

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update(self):
        self.iterations += 1
