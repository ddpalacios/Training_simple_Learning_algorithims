import numpy as np

class Adaline_gd(object):
	def __init__(self, lr= .01, epochs= 10):
		self.lr = lr
		self.epochs = epochs
	def fit(self, X,y):
		self.weights = np.random.rand(X.shape[1]+1)

		self.cost = []
		for _ in range(self.epochs):
			net = self.net_input(X)
			output = self.activate(net)
			error = (y - output)
			self.weights[1:] += self.lr * X.T.dot(error)
			self.weights[0] += self.lr * error.sum()
			cost = (-y.dot(np.log(output))- ((1-y).dot(np.log(1-output))))
			self.cost.append(cost)
		return self



	def net_input(self, X):
		return np.dot(X, self.weights[1:]) + self.weights[0]
	def activate(self, z):
		return 1 / (1+ np.exp(z))
	def predict(self, X):
		return np.where(self.activate(self.net_input(X)) >= 0.5, 1, 0)
	def score(self, X,y):
		prediction = self.predict(X)
		correct = np.count_nonzero(prediction ==y)
		print("Misclassified:", len(X)- correct)
		res = (correct/ len(X))*100
		mis = len(X) - correct
		wrong = (mis / len(X)) * 100
		return res, wrong


