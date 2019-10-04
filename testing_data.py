from ppn import Perceptron
from AdalineSGD import Adaline_sgd
from AdalineGD import Adaline_gd
from plot_regions import plot_decision_regions as plt_rgn
import matplotlib.pyplot as plt
import numpy as np
data_matrix = np.array([
       [1.2 ,1.1 ,1.0],
       [2.2, 5,  1.0],
       [1.0,   2,  1.0],
       [1.7 ,1.5, 1.0],
       [2.3, 4.1, 1.0],
       [4.0, 5.0, 0.0],
       [3,  9.0, 0.0],
       [4.5, 6.2, 0.0],
       [5.5 ,2.1 ,0.0],
       [7.7 ,5.4, 0.0 ]])

def column(matrix, i):
    return np.array([row[i] for row in matrix])


X = []
for i in range(len(data_matrix)):
    rows = data_matrix[i]
    X.append(rows[0:2])



#Usable features and labels
X = np.array(X)
target = column(data_matrix, 2)
# print("Data:\n{}\n\nTarget: {}".format(data,target))  #Set up features and labels

adaGD = Adaline_gd()
adaSGD = Adaline_sgd()
PPN = Perceptron()

adaGD.fit(X,target)
adaSGD.fit(X,target)
PPN.fit(X,target)

