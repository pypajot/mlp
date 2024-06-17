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
		self.activ_name = function
		self.neurons_base = np.empty((self.size, 1), float)
		self.neurons = np.empty((self.size, 1), float)
		self.activation = activation[function]
		self.deriv_acti = deriv_acti[function]


	def init_weights(self, prev_layer_size):
		self.weights = np.zeros((prev_layer_size, self.size), float)

	@property
	def activ_name(self):
		return self._activ_name

	@activ_name.setter
	def	activ_name(self, d):
		if d not in ['sigmoid', 'softmax', 'tanh', 'relu']:
			raise ValueError('activation function must be sigmoid, softmax, relu or tanh')
		self._activ_name = d
	
	@property
	def size(self):
		return self._size

	@size.setter
	def	size(self, s):
		if type(s) is not int or s <= 0:
			raise ValueError('layer size must be a positive integer')
		self._size = s