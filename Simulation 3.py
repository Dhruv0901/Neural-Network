import numpy as np
from nnfs.datasets import sine_data

from Adam_Optimiser import Optimiser_Adam
from Dense_Layer import Layer_Dense
from Linear_Activation import Activation_Linear
from MeanSquaredError_Loss import Loss_MeanSquaredError
from ReLU_Activation import Activation_ReLU

X,y = sine_data()

dense1 = Layer_Dense(1, 64) # the variance in weight initialisation is changed from 0.01 to 0.1
# this was done because during backpropagation the learning rate in adam does not have influence on the weights
# this results in the problem of gradient collapse as the gradients are too short to change anything
# this can also be achieved through Glorot or xavier distribution that adjusts itself based on number of inputs provided

activation1 = Activation_ReLU()

dense2 = Layer_Dense(64,64)

activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 1)

activation3 = Activation_Linear()

loss_function = Loss_MeanSquaredError()
optimiser = Optimiser_Adam(learning_rate=0.005, decay=1e-3)

accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)


    data_loss = loss_function.calculate(activation3.outputs, y)


    regularisation_loss = (loss_function.regularisation(dense1) +
                           loss_function.regularisation(dense2) +
                           loss_function.regularisation(dense3))

    loss = data_loss + regularisation_loss

    predictions = activation3.outputs
    accuracy = np.mean(np.absolute(predictions - y)<accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}'+
              f'acc: {accuracy}'+
              f'loss: {loss}')

    loss_function.backward(activation3.outputs, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinput)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimiser.pre_update_params()
    optimiser.update_params(dense1)
    optimiser.update_params(dense2)
    optimiser.update_params(dense3)
    optimiser.post_update()


import matplotlib.pyplot as plt

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.outputs)
plt.show()
