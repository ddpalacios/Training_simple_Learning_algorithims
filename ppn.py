#This script will demonstrate our simpliest classification model --> Perceptron
#Lets start by import the following libraries
import numpy as np
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
				update = self.predict(xi)
				error = self.lr * (target - update)  #The amount of error based on the target label
				#We will now use our error to update our weights accordinly
				self.weights[1:]+= error * xi
				self.weights[0]+= error

				cost += int(error !=0.0)  #If the error is NOT 0.0 then add it to the cost
			self.cost.append(cost) #and append it to our cost list so we can keep track of it
		return self


	def net_input(self,X):
		return np.dot(X, self.weights[1:]) + self.weights[0]

	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)

	def score(self, X,y):
		prediction = self.predict(X)
		correct = np.count_nonzero(prediction == y)
		print("Misclassified:", len(X) - correct)
		res = (correct / len(X)) *100
		mis = len(X) - correct
		wrong (mis / len(X))*100
		return res, wrong









