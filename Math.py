import math
import random

import numpy as np

def transpose(lst2):
    x = 0
    main_lst = []
    if x != len(lst2):
        for i in range(len(lst2[0])):
            lst = []
            for j in range(len(lst2)):
                lst.append(lst2[j][x])
            main_lst.append(lst)
            x += 1
        return main_lst
# time complexity of this algorithm is O(m * n) where m is the no of rows and n is the no of columns
def Multiplication(lst1, lst2):
    def lst_multiplication(l1, l2):
        a = 0
        for i in range(len(l1)):
            a += l1[i] * l2[i]
        var = a
        return var
    lst2 = transpose(lst2)
    main_lst = []
    for k in range(len(lst1)):
        ele = []
        for l in range(len(lst2)):
            ele.append(lst_multiplication(lst1[k], lst2[l]))
        main_lst.append(ele)
    return main_lst
# time complexity of this algorithm is O(m * n * p)
# w here m is the no of rows of matrix 1, n no of columns of matrix 2 and p no of rows of matrix 2
flag = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]
pointer = [
    [0.2, 0.5, -0.26],
    [0.8,-0.91,-0.27],
    [-0.5, 0.26, 0.17],
    [1, -0.5, 0.87]
]
print(Multiplication(flag, pointer))
def relu_activation(inputs):
    output = []
    for i in inputs:
        if i>0:
            output.append(i)
        else:
            output.append(0)
    return output
print(relu_activation([1, -2, 6, 8, -3]))


def max(lst):
    flag = lst[0]
    for i in lst:
        if i >= flag:
            flag = i
    return flag


def custom_sum(lst):
    total = 0
    for i in lst:
        total += i
    return total


def softmax_activation(inputs):
    e = 2.71828182846
    max_input = max(inputs)
    exp_values = []

    for i in inputs:
        exp_values.append(e ** (i - max_input))
    norm = []
    total_exp_sum = custom_sum(exp_values)
    for value in exp_values:
        norm.append(value / total_exp_sum)
    return norm


print(sum(softmax_activation([1, -2, 6, 8, -3])))# summation of all probabilites is 1 proving softmax activation's working

output = [0.56, 0.4, 0.04]
one_hot = [1, 0, 0]# can be interpreted as the output is index 0
loss = -(math.log(output[0])*one_hot[0]+
     math.log(output[1])*one_hot[1]+
     math.log(output[2])*one_hot[2])
print(loss)
loss = -(math.log(output[0]))# given the target index is 0 the formula can be re-written
print(loss)
outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])
class_targets = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]])
correct_confidences = np.sum(outputs*class_targets, axis=1)
# This is done for processing data in batches where class_targets and outputs are multiplied and summed together along each row

neg_log = -(np.log(correct_confidences))
avg_loss = np.mean(neg_log)
print(avg_loss)

# inputs = np.array([[1, 2, 3, 2.5],
#           [2, 5, -1, 2],
#           [-1.5, 2.7, 3.3, -0.8]])
# weights = np.array([[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]).T
# biases = [2, 3, 0.5]
# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]
#
# layer1_output = np.dot(inputs, np.array(weights).T) + biases # np.dot performs a dot product between two vectors and then np allows vector addition
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

# this is the foundational implementation of dropout layer where the output of certain nodes are switched-off
# to help the model better understand the data instead of generalisation

dropout_rate = 0.4
output = [-0.69, -0.26,  0.2 , -0.35, -0.26, -0.93, -0.73,  0.16, -0.74, -0.8 ]

while True:
    index = random.randint(0, len(output) - 1)
    output[index] = 0

    dropout = 0
    for i in output:
        if i==0:
            dropout+=1
    if dropout / len(output) >= dropout_rate:
        break

print(output)

print(np.random.binomial(1, 1-dropout_rate, 10))


