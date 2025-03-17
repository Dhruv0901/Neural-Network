from nnfs.datasets import spiral_data

from Adam_Optimiser import Optimiser_Adam
from CategoricalCrossentropy_Loss import Loss_CategoricalCrossentropy
from Categorical_Accuracy import Accuracy_Categorical
from Dense_Layer import Layer_Dense
from Dropout_Layer import Layer_Dropout
from Model import Model
from ReLU_Activation import Activation_ReLU
from Softmax_Activation import Activation_Softmax

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model1 = Model()
model1.add(Layer_Dense(2, 512, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))
model1.add(Activation_ReLU())
model1.add(Layer_Dropout(0.1))
model1.add(Layer_Dense(512, 3))
model1.add(Activation_Softmax())

model1.set(
    loss=Loss_CategoricalCrossentropy(),
    optimiser=Optimiser_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model1.finalise()
model1.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
