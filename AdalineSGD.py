import numpy as np
class Adaline_sgd(object):
	def __init__(self, lr=.01, epochs=10, shuffle=True, random_state=None):
		self.lr = lr
		self.epochs = epochs
		self.shuffle = shuffle
		self.weights_initialized = False
		self.random_state = random_state
	def fit(self, X,y):
		self.initalize_weights(X.shape[1])
		self.cost = []
		for _ in range(self.epochs):
			if self.shuffle:
				X,y = self.shuffle_data(X,y)
			cost = []
			for xi,target in zip(X,y):
				cost.append(self.update_weights(xi,target))
			avg_cost = sum(cost) / len(y)
			self.cost.append(avg_cost)
		return self

	def shuffle_data(self, X,y):
		r = self.rgen.permutaion(len(y))
		return X[r], y[r]
	def initalize_weights(self, m):
		self.rgen = np.random.RandomState(self.random_state)
		self.weights = self.rgen.normal(loc=0.0, scale=0.01, size= 1+m)
		self.weights_initialized = True
	def update_weights(self, xi,target):
		pass

	def net_input(self, X):
		pass

	def predict(self, X):
		pass
	def activation(self, X):
		pass

ada = Adaline_sgd()

print(ada)
