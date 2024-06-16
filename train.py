#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import argparse

from optimizers import optimizers, adam
from layer import Layer
from metrics import Metrics

# def get_part_deriv(mlp, delta, batch_index, i):
# 	if (i == mlp.size - 1):
# 		delta = mlp.layers[i].neurons - mlp.output[batch_index]
# 	else:
# 		delta = mlp.layers[i].deriv_acti(mlp.layers[i].neurons) * np.dot(delta, mlp.weights[i].T)
# 	return delta, np.mean(delta, 0), np.dot(mlp.layers[i - 1].neurons.T, delta) / len(batch_index)

# def adam(mlp, batch_index, steps):
# 	delta = 0
# 	for i in reversed(range(1, mlp.size)):
# 		delta, d_bias, d_weights = get_part_deriv(mlp, delta, batch_index, i)
# 		mlp.mt_b[i - 1] = mlp.beta1 * mlp.mt_b[i - 1] + (1 - mlp.beta1) * d_bias
# 		mlp.mt_w[i - 1] = mlp.beta1 * mlp.mt_w[i - 1] + (1 - mlp.beta1) * d_weights

# 		mlp.vt_b[i - 1] = mlp.beta2 * mlp.vt_b[i - 1] + (1 - mlp.beta2) * np.power(d_bias, 2)
# 		mlp.vt_w[i - 1] = mlp.beta2 * mlp.vt_w[i - 1] + (1 - mlp.beta2) * np.power(d_weights, 2)

# 		mt_corr_b = np.divide(mlp.mt_b[i - 1], 1 - np.power(mlp.beta1, steps))
# 		mt_corr_w = np.divide(mlp.mt_w[i - 1], 1 - np.power(mlp.beta1, steps))

# 		vt_corr_b = np.divide(mlp.vt_b[i - 1], 1 - np.power(mlp.beta2, steps))
# 		vt_corr_w = np.divide(mlp.vt_w[i - 1], 1 - np.power(mlp.beta2, steps))

# 		mt_corr_b = np.divide(mt_corr_b, np.sqrt(vt_corr_b) + mlp.epsilon)
# 		mt_corr_w = np.divide(mt_corr_w, np.sqrt(vt_corr_w) + mlp.epsilon)
# 		mlp.bias[i - 1] -= mlp.learning_rate * mt_corr_b
# 		mlp.weights[i - 1] -= mlp.learning_rate * mt_corr_w

# def gradient_descent(mlp, batch_index, steps):
# 	delta = 0
# 	for i in reversed(range(1, mlp.size)):
# 		delta, d_bias, d_weights = get_part_deriv(mlp, delta, batch_index, i)

# 		mlp.weights[i - 1] -= mlp.weights[i - 1] * mlp.regul

# 		mlp.velocity_w[i - 1] = mlp.momentum * mlp.velocity_w[i - 1] - mlp.learning_rate * d_weights
# 		mlp.velocity_b[i - 1] = mlp.momentum * mlp.velocity_b[i - 1] - mlp.learning_rate * d_bias
		
# 		mlp.weights[i - 1] += mlp.velocity_w[i - 1]
# 		mlp.bias[i - 1] += mlp.velocity_b[i - 1]
# 		# mlp.bias[i - 1] -= mlp.bias[i - 1] * mlp.regul / len(batch_index)


# optimizers = {
# 	'sgd': gradient_descent,
# 	'adam': adam
# }

# class Metrics:
# 	def	__init__(self):
# 		self.acc = 0
# 		self.loss = 0
# 		self.train_loss = []
# 		self.test_loss = []
# 		self.train_acc = []
# 		self.test_acc = []
# 		self.test_precision = 0
# 		self.test_recall = 0
# 		self.test_f1score = 0
# 		self.confusion_matrix = np.empty((0, 0))

# 	def get_test_loss_and_acc(self, mlp):
# 		loss = 0
# 		acc = 0
# 		mlp.feedforward(mlp.test_input)
# 		for i in range (mlp.test_output.size):
# 			loss -= math.log(mlp.layers[-1].neurons[i][mlp.test_output[i]])
# 			acc += mlp.test_output[i] == np.argmax(mlp.layers[-1].neurons[i])
# 		loss /= mlp.test_output.size
# 		acc /= mlp.test_output.size
# 		# for wei in mlp.weights:
# 		# 	loss += np.sum(np.power(wei, 2)) * mlp.regul * mlp.learning_rate
# 		self.test_loss.append(loss)
# 		self.test_acc.append(acc)

# 	def get_confusion_and_metrics(self, mlp):
# 		size = max(mlp.test_output) + 1
# 		self.confusion_matrix = np.zeros((size, size))
# 		self.acc = 0
# 		self.loss = 0
# 		for i in range (mlp.test_output.size):
# 			self.loss -= math.log(mlp.layers[mlp.size - 1].neurons[i][mlp.test_output[i]])
# 			self.confusion_matrix[mlp.test_output[i]][np.argmax(mlp.layers[mlp.size - 1].neurons[i])] += 1
# 			# true_pos += int(mlp.test_output[i] and mlp.test_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
# 			# true_neg += int((not mlp.test_output[i]) and mlp.test_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
# 			# false_pos += int(mlp.test_output[i] and mlp.test_output[i] != np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
# 			# false_neg += int((not mlp.test_output[i]) and mlp.test_output[i] != np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
# 		# acc = (true_pos + true_neg) / mlp.test_output.size
# 		if size == 2:
# 			self.acc = (self.confusion_matrix[0][0] + self.confusion_matrix[1][1]) / np.sum(self.confusion_matrix)
# 			self.precision = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])
# 			self.recall = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[0][1])
# 		else:
# 			for i in range(size):
# 				self.acc += self.confusion_matrix[i][i]
# 				self.precision += self.confusion_matrix[i][i] / (size * (np.sum(self.confusion_matrix[i][:])))
# 				self.recall += self.confusion_matrix[i][i] / (size * (np.sum(self.confusion_matrix[:][i])))
# 			self.acc /= np.sum(self.confusion_matrix)
# 		self.f1score = 2 * self.precision * self.recall / (self.precision + self.recall)


# 	def get_train_loss_and_acc(self, mlp, batch_index):
# 		for i in batch_index:
# 			# print(mlp.layers[mlp.size - 1].neurons[i])
# 			self.loss -= math.log(mlp.layers[mlp.size - 1].neurons[batch_index.index(i)][mlp.train_output[i]]) / mlp.train_output.size
# 			self.acc += (mlp.train_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[batch_index.index(i)])) / mlp.train_output.size
# 		# loss /= mlp.train_output.size
# 		# for wei in mlp.weights:
# 		# 	loss += np.sum(np.power(wei, 2)) * mlp.regul * mlp.learning_rate
# 		# self.train_loss.append(loss)
# 		# self.train_acc.append(acc)

# 	def add_loss_acc(self):
# 		self.train_loss.append(self.loss)
# 		self.train_acc.append(self.acc)

# 	def show(self):
# 		# fig, axs = plt.subplots(1, 2)
# 		axs[0].plot(range(len(self.train_loss)), self.train_loss, label='train')
# 		axs[0].plot(range(len(self.test_loss)), self.test_loss, label='validation')
# 		axs[1].plot(range(len(self.train_acc)), self.train_acc, label='train')
# 		axs[1].plot(range(len(self.test_acc)), self.test_acc, label='validation')
# 		axs[0].legend()
# 		axs[1].legend()
# 		fig.set_size_inches((15, 5))
# 		text_str = '\n'.join((
# 			'acc = {:.2}'.format(self.acc),
# 			'precision = {:.2}'.format(self.precision),
# 			'recall = {:.2}'.format(self.recall),
# 			'f1score = {:.2}'.format(self.f1score)
# 		))
# 		fig.text(0.92, 0.73, s=text_str)
# 		plt.show()

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
		distrib = 'LCuniform',
		momentum = 0.9,
		tol = 0.0001,
		n_iter_to_change = 10,
		early_stopping=False
	):
		# assert()
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
		self.metrics.show()

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


def main():

	parser = argparse.ArgumentParser(
		prog='train.py',
		description='train a MLPClassifier on some data'
	)
	parser.add_argument('data_train')
	parser.add_argument('data_test')
	parser.add_argument('-L', '--layers', type=tuple[int, ...], default=(100, 100))
	parser.add_argument('-e', '--epochs', type=int, default=300)
	parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
	parser.add_argument('-a', '--activation', default='relu')
	parser.add_argument('-b', '--batch_size', type=int, default=200)
	parser.add_argument('-o', '--optimizer', default='adam')
	parser.add_argument('-E', '--early_stopping', action='store_true')

	args = parser.parse_args()

	try:
		data_train = pd.read_csv(args.data_train, header = None)
		train_output = data_train[1]
		train_input = data_train.drop(columns=[0, 1])
	except Exception:
		print('training data is invalid')
		exit(2)


	try:
		data_test = pd.read_csv(args.data_test, header = None)
		test_ouput = data_test[1]
		test_input = data_test.drop(columns=[0, 1])
	except Exception:
		print('test data is invalid')
		exit(2)


	# from sklearn.neural_network import MLPClassifier

	# # fig, axs = plt.subplots(1, 2)
	# # clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=150, learning_rate_init=0.001, activation='relu', batch_size=200, solver='adam', early_stopping=True).fit(train_input, train_output)
	# # axs[0].plot(clf.loss_curve_, label='scikit')
	# # axs[1].plot(clf.validation_scores_, label='scikit')
	# # print(clf.validation_scores_)
	model = MultiLayerPerceptron(
		layers_sizes=args.layers,
		optimizer=args.optimizer,
		epochs=args.epochs,
		activation_func=args.activation,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		early_stopping=args.early_stopping
	)

	print(args.early_stopping)
	# mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'relu')
	# # mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'sigmoid')
	# mlp.add_layer(100, 'sigmoid')


	model.fit(train_input, train_output, test_input, test_ouput)

if __name__ == '__main__':
	main()
