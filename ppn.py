#This script will demonstrate our simpliest classification model --> Perceptron
#Lets start by import the following libraries and create a dummy dataset

import numpy as np
import matplotlib.pyplot as plt

data = np.array([
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
#Now that we have our Dataset, lets create our simple model
class Perceptron(object):
	def __init__(self, lr=.01, epochs=20):
		self.lr =lr
		self.epochs = epochs
	def fit(self, X,y):
		self.weights = np.random.rand(X.shape[1]+ 1)  #Creating our starting weights with a bias
		np.seed(0) #To make sure there are no different weight values

		self.cost = []  #We need to keep track of our learning expenses so we can graph later


		for _ in range(self.epochs):
			error = 0 #Amount of errors per epochs

			for xi, target in zip(X,y):
				update = self.lr * (self.predict(xi))

	def net_input(self,X):
		pass

	def predict(self, X):
		pass

	def score(self, X):
		pass




ppn = Perceptron()

print(ppn)


