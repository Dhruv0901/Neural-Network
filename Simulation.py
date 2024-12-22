from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np

from Adam_Optimiser import Optimiser_Adam
from CrossEntropyLoss_Softmax_Activtion import Activation_Softmax_Loss_CategoricalCrossEntropy
from Dense_Layer import Layer_Dense
from ReLU_Activation import Activation_ReLU
from Softmax_Activation import Activation_Softmax
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

loss_function = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimiser = Optimiser_Adam(learning_rate=0.05, decay=5e-7)


for epoch in range(10001):
    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_function.forward(dense2.output, y)

    predictions = np.argmax(loss_function.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f"epoch: ", {epoch},
              f"accuracy: ", {accuracy},
              f"loss: ", {loss},
              f"learning rate ", {optimiser.learning_rate})

    loss_function.backward(loss_function.output, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimiser.pre_update_params()
    optimiser.update_params(dense1)
    optimiser.update_params(dense2)
    optimiser.post_update()

