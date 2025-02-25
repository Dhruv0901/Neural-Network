import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Adam_Optimiser import Optimiser_Adam
from BinaryCrossentropy_Loss import Loss_BinaryCrossentropy
from Dense_Layer import Layer_Dense
from ReLU_Activation import Activation_ReLU
from Sigmoid_Activation import Activation_Sigmoid
from Simulation import activation1, loss_function, predictions, data_loss, regularisation_loss, accuracy

nnfs.init()

X,y = spiral_data(100, 2)

y = y.reshape(-1,1)

dense1 = Layer_Dense(2,64, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 1)
activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()
optimiser = Optimiser_Adam(decay=5e-7)


# for epoch in range(10001):
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#
#     activation2.forward(dense2.output)
#     data_loss = loss_function.calculate(activation2.output, y)
#
#     regularisation_loss = (loss_function.regularisation_loss(dense1)
#                            + loss_function.regularisation(dense2))
#
#     loss = data_loss + regularisation_loss
#
#     predictions = (activation2.output > 0.5) * 1
#     accuracy = np.mean(predictions==y)
#
#     if not epoch % 100:
#         print(f"epoch: ", {epoch},
#               f"accuracy: ", {accuracy},
#               f"loss: ", {loss},
#               f"learning rate ", {optimiser.current_learning_rate})
#
#     loss_function.backward(loss_function.output, y)
#     activation2.backward(loss_function.dinputs)
#     dense2.backward(loss_function.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)
#
#     optimiser.pre_update_params()
#     optimiser.update_params(dense1)
#     optimiser.update_params(dense2)
#     optimiser.post_update()

X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
