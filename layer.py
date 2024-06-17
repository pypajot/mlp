import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def	sigmoid_der(x):
	return x * (1 - x)

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), 1)[:, np.newaxis]

def	softmax_der(x):
	return 0

def tanh(x):
	return np.tanh(x)

def	tanh_der(x):
	return 1 - x ** 2

def	relu(x):
	return x * (x > 0)

def	relu_der(x):
	return x > 0

activation = {
	'sigmoid': sigmoid,
	'softmax': softmax,
	'tanh': tanh,
	'relu': relu,
}

deriv_acti = {
	'sigmoid': sigmoid_der,
	'softmax': softmax_der,
	'tanh': tanh_der,
	'relu': relu_der
}

class Layer:
	def	__init__(self, size, function = 'relu'):
		self.size = size
		self.neurons_base = np.empty((self.size, 1), float)
		self.neurons = np.empty((self.size, 1), float)
		self.activation = activation[function]
		self.deriv_acti = deriv_acti[function]


	def init_weights(self, prev_layer_size):
		self.weights = np.zeros((prev_layer_size, self.size), float)