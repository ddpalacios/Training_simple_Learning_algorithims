import numpy as np
class Adaline_sgd(object):
	def __init__(self, lr=.01, epochs=10, shuffle=True, random_state=None):
		self.lr = lr
		self.epochs = epochs
		self.shuffle = shuffle
		self.weights_initialized = False
		self.random_state = random_state

	def fit(self, X,y):
		self.initialize_weights(X.shape[1])
		self.cost = []
		for _ in range(self.epochs):

			#Be sure to shuffle our data for performance
			if self.shuffle:
				X,y = self.shuffle_data(X,y)



			cost = []
			for xi,target in zip(X,y):
				cost.append(self.update_weights(xi,target))
			avg_cost = sum(cost) / len(y)
			self.cost.append(avg_cost)
		return self





	def shuffle_data(self, X,y):
		r = self.rgen.permutation(len(y))
		return X[r], y[r]
	def initialize_weights(self, m):
		self.rgen = np.random.RandomState(self.random_state)
		self.weights = self.rgen.normal(loc=0.0, scale=0.01, size= 1+m)
		self.weights_initialized = True
	def update_weights(self, xi,target):
		#Lets apply to adaline learning rule to update our weights
		net = self.net_input(xi)
		output = self.activate(net)
		error = (target - output)
		self.weights[1:]+= self.lr * xi.dot(error)
		self.weights[0]+= self.lr * error
		cost = (-target.dot(np.log(output))- ((1-target).dot(np.log(1-output))))
		return cost

	def net_input(self, X):
		return np.dot(X, self.weights[1:])+ self.weights[0]
	def predict(self, X):
		return np.where(self.activate(self.net_input(X)) >= 0.5, 1,0)
	def activate(self, z):
		return 1/ (1 + np.exp(-z))

