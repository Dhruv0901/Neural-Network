from nnfs.datasets import spiral_data
import numpy as np
from Dense_Layer import Layer_Dense


inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer1_output = np.dot(inputs, np.array(weights).T) + biases # np.dot performs a dot product between two vectors and then np allows vector addition
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

X,y = spiral_data(100, 3)# spiral data of two points x and y stored as X and three classes 0, 1 and 2
dense1 = Layer_Dense(2,3)# two inputs(x and y of the points) and three neurons
dense1.forward(X)# values stored in X passed so a list with two elements are multiplied with the 3 neurons
print(dense1.output[:7])

