import numpy as np


class Optimiser_RMSprop:

    def __init__(self, learning_rate=0.001, decay=0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):# step learning decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # cache just like Adagrad but instead of epsilon as the hyperparameter for cache
        # we use rho which is the cache memory decay because even with simplistic model
        # it will carry much momentum that even with smaller updates it will keep on going
        if not hasattr(layer, 'weight_cache'):# initialises an array if it does not exist before
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        # still used epsilon for the usual vanilla gradient
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update(self):
        self.iterations += 1
