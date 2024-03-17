#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

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
		self.train_loss.append(loss / mlp.train_output.size)
		self.train_acc.append(acc / mlp.train_output.size)
		loss = 0
		acc = 0
		mlp.feedforward(mlp.test_input)
		for i in range (mlp.test_output.size):
			loss -= math.log(mlp.layers[mlp.size - 1].neurons[i][mlp.test_output[i]])
			acc += (mlp.test_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
		self.test_loss.append(loss / mlp.test_output.size)
		self.test_acc.append(acc / mlp.test_output.size)

	def show_curves(self):
		fig, axs = plt.subplots(1, 2)
		axs[0].plot(range(len(self.train_loss)), self.train_loss)
		axs[0].plot(range(len(self.test_loss)), self.test_loss)
		axs[1].plot(range(len(self.train_acc)), self.train_acc)
		axs[1].plot(range(len(self.test_acc)), self.test_acc)
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
		'Relu': (lambda x: x*(x > 0)),
	}

	deriv_acti = {
		'sigmoid': sigmoid_der,
		'softmax': (lambda x: 0),
		'tanh': (lambda x: 1-x**2),
		'Relu': (lambda x: 1 * (x>0))
	}

	def init_weights(self, prev_layer_size):
		self.weights = np.zeros((prev_layer_size, self.size), float)
		

class MultiLayerPerceptron:
	def __init__(self, epochs = 200, learning_rate = 0.001, batch_size = 1, seed = None):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.size = 2
		self.layers_sizes = []
		self.layers = []
		self.weights = []
		self.bias = []
		self.rng = np.random.default_rng(seed)
		
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
			weights = np.array(self.rng.uniform(-1, 1, (self.layers_sizes[i - 1], self.layers_sizes[i])))
			self.weights.append(weights)
			bias = np.array(self.rng.uniform(-1, 1, (1, self.layers_sizes[i])))
			self.bias.append(bias)

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
		for i in range (self.epochs):
			batch_set = batch_init(self.input.shape[0])
			while (batch_set):
				batch_index = get_batch(batch_set, self.batch_size)
				input = self.input[batch_index]
				self.feedforward(input)
				self.__backprop(batch_index)
			curves.get_loss_and_acc(self)
		curves.show_curves()

	def feedforward(self, input):
		self.layers[0].neurons = input
		for i in range (1, self.size):
			self.layers[i].neurons_base = np.dot(self.layers[i - 1].neurons, self.weights[i - 1]) + self.bias[i - 1]
			self.layers[i].neurons = self.layers[i].activation(self.layers[i].neurons_base)
		return self.layers[self.size - 1].neurons

	def	__backprop(self, batch_index):
		for i in reversed(range(1, self.size)):
			if (i == self.size - 1):
				delta = self.layers[i].neurons - self.output[batch_index]
			else:
				delta = self.layers[i].deriv_acti(self.layers[i].neurons) * np.dot(delta, self.weights[i].T)
			self.bias[i - 1] -= self.learning_rate / len(batch_index) * np.sum(delta, 0)
			self.weights[i - 1] -= self.learning_rate / len(batch_index) * np.dot(self.layers[i - 1].neurons.T, delta)
			
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

# clf = MLPClassifier(random_state=1, max_iter=300, learning_rate_init=0.1, activation='logistic', batch_size=1, solver='sgd').fit(train_input, train_output)
# plt.plot(clf.loss_curve_)
# plt.show()

mlp = MultiLayerPerceptron()
mlp.add_layer(100, 'sigmoid')
mlp.add_layer(100, 'sigmoid')
mlp.add_layer(100, 'sigmoid')
# mlp.add_layer(100, 'sigmoid')


mlp.fit(train_input, train_output, test_input, test_ouput)
