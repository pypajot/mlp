#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

fig, axs = plt.subplots(1, 2)

def get_part_deriv(mlp, delta, batch_index, i):
	if (i == mlp.size - 1):
		delta = mlp.layers[i].neurons - mlp.output[batch_index]
	else:
		delta = mlp.layers[i].deriv_acti(mlp.layers[i].neurons) * np.dot(delta, mlp.weights[i].T)
	return delta, np.mean(delta, 0), np.dot(mlp.layers[i - 1].neurons.T, delta) / len(batch_index)

def adam(mlp, batch_index, steps):
	delta = 0
	for i in reversed(range(1, mlp.size)):
		delta, d_bias, d_weights = get_part_deriv(mlp, delta, batch_index, i)
		mlp.mt_b[i - 1] = mlp.beta1 * mlp.mt_b[i - 1] + (1 - mlp.beta1) * d_bias
		mlp.mt_w[i - 1] = mlp.beta1 * mlp.mt_w[i - 1] + (1 - mlp.beta1) * d_weights

		mlp.vt_b[i - 1] = mlp.beta2 * mlp.vt_b[i - 1] + (1 - mlp.beta2) * np.power(d_bias, 2)
		mlp.vt_w[i - 1] = mlp.beta2 * mlp.vt_w[i - 1] + (1 - mlp.beta2) * np.power(d_weights, 2)

		mt_corr_b = np.divide(mlp.mt_b[i - 1], 1 - np.power(mlp.beta1, steps))
		mt_corr_w = np.divide(mlp.mt_w[i - 1], 1 - np.power(mlp.beta1, steps))

		vt_corr_b = np.divide(mlp.vt_b[i - 1], 1 - np.power(mlp.beta2, steps))
		vt_corr_w = np.divide(mlp.vt_w[i - 1], 1 - np.power(mlp.beta2, steps))

		mt_corr_b = np.divide(mt_corr_b, np.sqrt(vt_corr_b) + mlp.epsilon)
		mt_corr_w = np.divide(mt_corr_w, np.sqrt(vt_corr_w) + mlp.epsilon)
		mlp.bias[i - 1] -= mlp.learning_rate * mt_corr_b
		mlp.weights[i - 1] -= mlp.learning_rate * mt_corr_w

def gradient_descent(mlp, batch_index, steps):
	delta = 0
	for i in reversed(range(1, mlp.size)):
		delta, d_bias, d_weights = get_part_deriv(mlp, delta, batch_index, i)

		mlp.weights[i - 1] -= mlp.weights[i - 1] * mlp.regul

		mlp.velocity_w[i - 1] = mlp.momentum * mlp.velocity_w[i - 1] - mlp.learning_rate * d_weights
		mlp.velocity_b[i - 1] = mlp.momentum * mlp.velocity_b[i - 1] - mlp.learning_rate * d_bias
		
		mlp.weights[i - 1] += mlp.velocity_w[i - 1]
		mlp.bias[i - 1] += mlp.velocity_b[i - 1]
		# mlp.bias[i - 1] -= mlp.bias[i - 1] * mlp.regul / len(batch_index)


optimizers = {
	'sgd': gradient_descent,
	'adam': adam
}

class LossAcc:
	def	__init__(self):
		self.train_loss = []
		self.test_loss = []
		self.train_acc = []
		self.test_acc = []

	def get_loss_and_acc(self, mlp):
		mlp.feedforward(mlp.input)
		loss = 0
		acc = 0
		for i in range (mlp.train_output.size):
			loss -= math.log(mlp.layers[mlp.size - 1].neurons[i][mlp.train_output[i]])
			acc += (mlp.train_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
		loss /= mlp.train_output.size
		# for wei in mlp.weights:
		# 	loss += np.sum(np.power(wei, 2)) * mlp.regul * mlp.learning_rate
		self.train_loss.append(loss)
		self.train_acc.append(acc / mlp.train_output.size)
		loss = 0
		acc = 0
		mlp.feedforward(mlp.test_input)
		for i in range (mlp.test_output.size):
			loss -= math.log(mlp.layers[mlp.size - 1].neurons[i][mlp.test_output[i]])
			acc += (mlp.test_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
		loss /= mlp.test_output.size
		# for wei in mlp.weights:
		# 	loss += np.sum(np.power(wei, 2)) * mlp.regul * mlp.learning_rate
		self.test_loss.append(loss)
		self.test_acc.append(acc / mlp.test_output.size)

	def show_curves(self):
		# fig, axs = plt.subplots(1, 2)
		axs[0].plot(range(len(self.train_loss)), self.train_loss, label='train')
		axs[0].plot(range(len(self.test_loss)), self.test_loss, label='validation')
		axs[1].plot(range(len(self.train_acc)), self.train_acc)
		axs[1].plot(range(len(self.test_acc)), self.test_acc)
		axs[0].legend()
		fig.set_size_inches((12, 5))
		plt.show()

def batch_init(size):
	set = list(range(size))
	np.random.shuffle(set)
	return set

def	get_batch(set, size):
	size = min(size, len(set))
	batch = set[:size]
	del set[:size]
	return batch

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def	sigmoid_der(x):
	return x * (1 - x)

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), 1)[:, np.newaxis]

class Layer:
	def	__init__(self, size, function = 'sigmoid'):
		self.size = size
		self.neurons_base = np.empty((self.size, 1), float)
		self.neurons = np.empty((self.size, 1), float)
		self.activation = self.activation[function]
		self.deriv_acti = self.deriv_acti[function]

	activation = {
		'sigmoid': sigmoid,
		'softmax': softmax,
		'tanh': (lambda x: np.tanh(x)),
		'relu': (lambda x: x * (x > 0)),
	}

	deriv_acti = {
		'sigmoid': sigmoid_der,
		'softmax': (lambda x: 0),
		'tanh': (lambda x: 1 - x**2),
		'relu': (lambda x: (x > 0))
	}

	def init_weights(self, prev_layer_size):
		self.weights = np.zeros((prev_layer_size, self.size), float)
		

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
	def __init__(self, epochs = 300, learning_rate = 0.01, batch_size = 1, optimizer = 'sgd', regul = 0.9, seed = None, distrib = 'LCuniform', momentum = 0.9):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
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

	def	__init_data(self, train_input, train_output, test_input, test_output):
		self.input = np.array(train_input)
		size_input = train_input.shape[1]
		self.layers.insert(0, Layer(size_input))
		self.layers_sizes.insert(0, size_input)

		self.train_output = np.array(train_output)
		self.output = np.zeros((train_output.size, train_output.max() + 1))
		self.output[np.arange(train_output.size), train_output] = 1
		self.possible_output = np.unique(train_output)
		size_output = len(self.possible_output)
		self.layers.append(Layer(size_output, 'softmax'))
		self.layers_sizes.append(size_output)

		self.test_input = test_input
		self.test_output = test_output

	def	__init_weights(self):
		for i in range (1, len(self.layers_sizes)):
			if self.optimizer == adam:
				self.mt_b.append(np.zeros((1, self.layers_sizes[i])))
				self.vt_b.append(np.zeros((1, self.layers_sizes[i])))
				self.mt_w.append(np.zeros((self.layers_sizes[i - 1], self.layers_sizes[i])))
				self.vt_w.append(np.zeros((self.layers_sizes[i - 1], self.layers_sizes[i])))
			lim = np.sqrt(6 / (self.layers_sizes[i - 1] + self.layers_sizes[i]))
			weights = np.array(self.rng.uniform(-lim, lim, (self.layers_sizes[i - 1], self.layers_sizes[i])))
			self.weights.append(weights)
			bias = np.array(self.rng.uniform(-lim, lim, (1, self.layers_sizes[i])))
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

	def	__train(self):
		curves = LossAcc()
		steps = 0
		for i in range (self.epochs):
			batch_set = batch_init(self.input.shape[0])
			while (batch_set):
				batch_index = get_batch(batch_set, self.batch_size)
				input = self.input[batch_index]
				self.feedforward(input)
				steps += 1
				self.__backprop(batch_index, steps)
			curves.get_loss_and_acc(self)
		curves.show_curves()

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


data_train = pd.read_csv('data_train.csv', header = None)
train_output = data_train[1]
train_input = data_train.drop(columns=[0, 1])

data_test = pd.read_csv('data_test.csv', header = None)
test_ouput = data_test[1]
test_input = data_test.drop(columns=[0, 1])

from sklearn.neural_network import MLPClassifier

# fig, axs = plt.subplots(1, 2)
clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=40, learning_rate_init=0.001, activation='relu', batch_size=10, solver='sgd', early_stopping=True).fit(train_input, train_output)
axs[0].plot(clf.loss_curve_, label='scikit')

mlp = MultiLayerPerceptron(optimizer='sgd', epochs=40, batch_size=10, learning_rate=0.001, regul = 0.0001)
mlp.add_layer(100, 'relu')
mlp.add_layer(100, 'relu')
mlp.add_layer(100, 'relu')
mlp.add_layer(100, 'relu')
# mlp.add_layer(100, 'relu')
# mlp.add_layer(100, 'sigmoid')
# mlp.add_layer(100, 'sigmoid')


mlp.fit(train_input, train_output, test_input, test_ouput)
