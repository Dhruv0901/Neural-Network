import numpy as np


class Optimiser_SGD():

    def __init__(self, learning_rate=1.0, decay=0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):# step learning decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # momentum just like real world is like an extra force that keeps the optimiser moving so it
        # does not get stuck within a local minimum
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):# initialises an array if it does not exist before
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.weights
            layer.weight_momentums = weight_updates# layer's weight momentum is updated
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.biases
            layer.bias_momentums = bias_updates# layer's bias momentum is updated
        else:
            # normal vanilla optimisation
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update(self):
        self.iterations += 1

    # The role of an optimiser is to find the global minimum of a loss function f(x) with respect to the
    # variable x of the lost function if taken unilaterally