import numpy as np
import random

from optimizers import optimizers, adam
from layer import Layer
from metrics import Metrics

def batch_init(size):
	set = list(range(size))
	np.random.shuffle(set)
	return set

def	get_batch(set, size):
	size = min(size, len(set))
	batch = set[:size]
	del set[:size]
	return batch
		

def normal(rng, F_in, F_out):
	limit = 0.4
	return rng.normal(0.0, limit, (F_in, F_out))

def uniform(rng, F_in, F_out):
	limit = 0.2
	return rng.uniform(-limit, limit, (F_in, F_out))

def	LCnormal(rng, F_in, F_out):
	limit = np.sqrt(1 / F_in)
	return rng.normal(0.0, limit, (F_in, F_out))

def LCuniform(rng, F_in, F_out):
	limit = np.sqrt(3 / F_in)
	return rng.uniform(-limit, limit, (F_in, F_out))

def	XGnormal(rng, F_in, F_out):
	limit = np.sqrt(2 / (F_in + F_out))
	return rng.normal(0.0, limit, (F_in, F_out))

def XGuniform(rng, F_in, F_out):
	limit = np.sqrt(6 / (F_in + F_out))
	return rng.uniform(-limit, limit, (F_in, F_out))

def	Henormal(rng, F_in, F_out):
	limit = np.sqrt(2 / F_in)
	return rng.normal(0.0, limit, (F_in, F_out))

def Heuniform(rng, F_in, F_out):
	limit = np.sqrt(6 / F_in)
	return rng.uniform(-limit, limit, (F_in, F_out))

distribution = {
		'normal': normal,
		'uniform': uniform,
		'LCnormal': LCnormal,
		'LCuniform': LCuniform,
		'XGnormal': XGnormal,
		'XGuniform': XGuniform,
		'Henormal': Henormal,
		'Heuniform': Heuniform,

}

class MultiLayerPerceptron:
	def __init__(
		self,
		layers_sizes=(100, 100),
		activation_func='relu',
		epochs = 300,
		learning_rate = 0.01,
		batch_size = 1,
		optimizer = 'sgd',
		regul = 0.0001,
		seed = None,
		distrib = 'XGuniform',
		output_layer_activation='softmax',
		momentum = 0.9,
		tol = 0.0001,
		n_iter_to_change = 10,
		early_stopping=False
	):
		# assert()
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.output_layer_activation = output_layer_activation
		self.size = 2
		self.layers_sizes = []
		self.layers = []
		self.weights = []
		self.bias = []
		self.optimizer = optimizers[optimizer]
		self.mt_b = []
		self.mt_w = []
		self.vt_b = []
		self.vt_w = []
		self.velocity_w = []
		self.velocity_b = []
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.regul = regul
		self.epsilon = 1e-8
		self.rng = np.random.default_rng(seed)
		self.distrib = distribution[distrib]
		self.momentum = momentum
		self.tol = tol
		self.n_iter_to_change = n_iter_to_change
		self.early_stopping = early_stopping
		self.best_loss = np.inf
		self.best_acc = 0
		for size in layers_sizes:
			self.add_layer(size, activation_func)

	def	__init_data(self, train_input, train_output, test_input, test_output):
		self.input = np.array(train_input)
		self.size_input = train_input.shape[1]
		self.layers.insert(0, Layer(self.size_input))
		self.layers_sizes.insert(0, self.size_input)

		self.train_output = np.array(train_output)
		self.output = np.zeros((train_output.size, train_output.max() + 1))
		self.output[np.arange(train_output.size), train_output] = 1
		if self.output_layer_activation == 'softmax':
			self.possible_output = np.unique(train_output)
			self.size_output = len(self.possible_output)
		else:
			self.size_output = 1
		self.layers.append(Layer(self.size_output, self.output_layer_activation))
		self.layers_sizes.append(self.size_output)

		self.test_input = test_input
		self.test_output = test_output

	def	__init_weights(self):
		for i in range (1, len(self.layers_sizes)):
			if self.optimizer == adam:
				self.mt_b.append(np.zeros((1, self.layers_sizes[i])))
				self.vt_b.append(np.zeros((1, self.layers_sizes[i])))
				self.mt_w.append(np.zeros((self.layers_sizes[i - 1], self.layers_sizes[i])))
				self.vt_w.append(np.zeros((self.layers_sizes[i - 1], self.layers_sizes[i])))
			weights = np.array(self.distrib(self.rng, self.layers_sizes[i - 1], self.layers_sizes[i]))
			self.weights.append(weights)
			bias = np.array(self.distrib(self.rng, 1, self.layers_sizes[i]))
			self.bias.append(bias)
			if self.momentum != 0:
				self.velocity_w.append(np.zeros(weights.shape))
				self.velocity_b.append(np.zeros(bias.shape))

	def	add_layer(self, size, function):
		self.layers.append(Layer(size, function))
		self.layers_sizes.append(size)
		self.size += 1

	def fit(self, train_input, train_output, test_input, test_output):
		self.__init_data(train_input, train_output, test_input, test_output)
		self.__init_weights()
		self.__train()

	def check_early_stopping(self, metrics: Metrics, no_changes, i):
		# if self.early_stopping == False:
		# 	return 0
		if self.early_stopping:
			if metrics.test_acc[-1] - self.best_acc < self.tol:
				no_changes += 1
			else:
				no_changes = 0
			if self.best_acc <= metrics.test_acc[-1]:
				self.best_acc = metrics.test_acc[-1]
				self.best_epoch = i
				self.best_weights = [w.copy() for w in self.weights]
				self.best_bias = [b.copy() for b in self.bias]
		else:
			if self.best_loss - metrics.train_loss[-1] < self.tol:
				no_changes += 1
			else:
				no_changes = 0
			if self.best_loss >= metrics.train_loss[-1]:
				self.best_loss = metrics.train_loss[-1]
		return no_changes

	def	__train(self):
		self.metrics = Metrics()
		# print(curves.acc)
		steps = 0
		no_changes = 0
		self.batch_size = min(self.batch_size, self.size_input)
		for i in range (self.epochs):
			batch_set = batch_init(self.input.shape[0])
			self.metrics.loss = 0
			self.metrics.acc = 0
			while (batch_set):
				batch_index = get_batch(batch_set, self.batch_size)
				input = self.input[batch_index]
				# output = self.train_output[batch_index]
				self.feedforward(input)
				self.metrics.get_train_loss_and_acc(self, batch_index)
				steps += 1
				self.__backprop(batch_index, steps)
			self.metrics.add_loss_acc()
			self.metrics.get_test_loss_and_acc(self)
			print('epochs: {}/{} - loss: {:.4} - val_loss: {:.4}'.format(i, self.epochs, self.metrics.train_loss[-1], self.metrics.test_loss[-1]))
			no_changes = self.check_early_stopping(self.metrics, no_changes, i)
			if no_changes >= self.n_iter_to_change:
				print('Stopped early at epoch', self.best_epoch if self.early_stopping else i)
				if self.early_stopping:
					self.weights = [w.copy() for w in self.best_weights]
					self.bias = [b.copy() for b in self.best_bias]
					self.feedforward(self.test_input)
				break
			# curves.test_loss = curves.test_loss[:self.best_step]
			# curves.train_loss = curves.train_loss[:self.best_step]
			# curves.test_acc = curves.test_acc[:self.best_step]
			# curves.train_acc = curves.train_acc[:self.best_step]
		self.metrics.get_confusion_and_metrics(self)
		# self.curves.show_curves()

	def feedforward(self, input):
		self.layers[0].neurons = input
		for i in range (1, self.size):
			self.layers[i].neurons_base = np.dot(self.layers[i - 1].neurons, self.weights[i - 1]) + self.bias[i - 1]
			self.layers[i].neurons = self.layers[i].activation(self.layers[i].neurons_base)
		return self.layers[self.size - 1].neurons

	def	__backprop(self, batch_index, steps):
		self.optimizer(self, batch_index, steps)
		# for i in reversed(range(1, self.size)):
		# 	if (i == self.size - 1):
		# 		delta = self.layers[i].neurons - self.output[batch_index]
		# 	else:
		# 		delta = self.layers[i].deriv_acti(self.layers[i].neurons) * np.dot(delta, self.weights[i].T)
		# 	self.bias[i - 1] -= self.learning_rate / len(batch_index) * np.sum(delta, 0)
		# 	self.weights[i - 1] -= self.learning_rate / len(batch_index) * np.dot(self.layers[i - 1].neurons.T, delta)
			
	# def test(self, test_input, test_output):
	# 	result = self.feedforward(test_input)
	# 	# print(result)
	# 	# for weights in self.weights:
	# 	print(self.weights[0])
