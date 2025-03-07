import numpy as np


class Optimiser_Adam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):# step learning decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # Adam works on top of RMSP when it comes to cache decay and combines it with SGD with momentum
        # This allows for both gradient and momentum fixing
        if not hasattr(layer, 'weight_cache'):# initialises an array if it does not exist before
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # calculates current momentum using the formula for Root Mean Square Propagation
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # corrected momentum is momentum divided by value approaching 1
        # this is done by 1 - lim(step ->0/0) 0.9 ^ step
        # this means as the step reaches infinity beta_1 tends to reach 0 thus 1-0 reaches 1
        weight_momentum_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # calculates cache just like RMSProp
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # correction just like momentum
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # vanilla SGD normalised through squared cache
        layer.weights += -self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update(self):
        self.iterations += 1
