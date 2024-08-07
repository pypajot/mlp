import numpy as np
import pandas as pd

from modules.optimizers import optimizers
from modules.layer import Layer
from modules.metrics import Metrics
from modules.distribution import distribution
from modules.utils import batch_init, get_batch
from seperate import seperate

class MultiLayerPerceptronClassifier:
	"""Multi-layer perceptron classifier

	Parameters:
		layers_sizes: tuple, optional
			Number of neurons in each hidden layer
			Defaults to (100, 100)
		activation_func: {"relu", "sigmoid", "tanh"}, optional
			Activation function of hidden layers
			Defaults to 'relu'
		epochs: int, optional
			Maximum umber of epochs
			Defaults to 300
		learning_rate: float, optional
			Learning rate
			Defaults to 0.01
		batch_size: int, optional
			Batch size
			Defaults to 200
		optimizer: {"sgd", "adam"}, optional
			Stochastic optimizer
			Defaults to 'adam'
		regul: float, optional
			L2 regularization term
			Defaults to 0.0001
		seed: int, optional
			Random seed for weights initialization
			Defaults to None
		distrib: {'normal', 'uniform', 'LCnormal', 'LCuniform', 'XGnormal', 'XGuniform', 'Henormal', 'Heuniform'}, optional
			Weight initialization distribution
			Defaults to 'XGuniform'
		output_layer_activation: {"softmax", "sigmoid"}, optional
			Output layer activation function
			Defaults to 'softmax'
		momentum: float, optional
			Momentum Only used with solver 'sgd'
			Defaults to 0.9
		nesterov: bool, optional
			Whether to use nesterov momentum
			Only used with solver 'sgd'
			Defaults to False
		tol: float, optional
			Difference needed for loss or accuracy during n_iter_to_change epochs to trigger stopping
			Defaults to 0.0001
		n_iter_to_change: int, optional
			Number of iterations to change
			Defaults to 10
		early_stopping: bool, optional
			Whether to use early stopping
			If true dataset will be split into train and test set according to split, and stopping will use accuracy
			Defaults to False
		beta1: float, optional
			Beta1 term for adam optimizer
			Defaults to 0.9
		beta2: float, optional
			Beta2 term for adam optimizer
			Defaults to 0.999
		name: str, optional
			Name for comparison with other models
			Defaults to None
		split: float, optional
			Split for train and test set if early_stopping is used
			Defaults to 0.8
	"""

	def __init__(
		self,
		layers_sizes=(100, 100),
		activation_func='relu',
		epochs=300,
		learning_rate=0.01,
		batch_size=200,
		optimizer='adam',
		regul=0.0001,
		seed=None,
		distrib='XGuniform',
		output_layer_activation='softmax',
		momentum=0.9,
		nesterov=False,
		tol=0.0001,
		n_iter_to_change=10,
		early_stopping=False,
		beta1=0.9,
		beta2=0.999,
		name=None,
		split=0.8
	):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.output_layer_activation = output_layer_activation
		self.momentum = momentum
		self.nesterov = nesterov
		self.activation_func = activation_func
		self.regul = regul
		self.tol = tol
		self.n_iter_to_change = n_iter_to_change
		self.early_stopping = early_stopping
		self.distrib_name = distrib
		self.opti_name = optimizer
		self.beta1 = beta1
		self.beta2 = beta2
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
		self.epsilon = 1e-8
		self.distrib = distribution[distrib]
		self.best_loss = np.inf
		self.best_acc = 0
		self.converged_in = 0
		self.seed=seed
		self.rng = np.random.default_rng(seed)
		self.name = name
		self.split = split
		for size in layers_sizes:
			self.add_layer(size, activation_func)


	@property
	def epochs(self):
		return self._epochs

	@epochs.setter
	def	epochs(self, e):
		if type(e) is not int or e <= 0:
			raise ValueError('epoch must be a positive integer')
		self._epochs = e
	
	@property
	def learning_rate(self):
		return self._learning_rate

	@learning_rate.setter
	def	learning_rate(self, e):
		if e <= 0:
			raise ValueError('learning rate must be positive')
		self._learning_rate = e
	
	@property
	def batch_size(self):
		return self._batch_size

	@batch_size.setter
	def	batch_size(self, e):
		if type(e) is not int or e <= 0:
			raise ValueError('batch size must be a positive integer')
		self._batch_size = e
	
	@property
	def output_layer_activation(self):
		return self._output_layer_activation

	@output_layer_activation.setter
	def	output_layer_activation(self, e):
		if e not in ['softmax', 'sigmoid']:
			raise ValueError('activation function must be softmax or sigmoid')
		self._output_layer_activation = e
	
	@property
	def activation_func(self):
		return self._activation_func

	@activation_func.setter
	def	activation_func(self, e):
		if e not in ['sigmoid', 'relu', 'tanh']:
			raise ValueError('activation function must be relu, sigmoid or tanh')
		self._activation_func = e

	@property
	def momentum(self):
		return self._momentum

	@momentum.setter
	def	momentum(self, e):
		if e < 0 or e > 1:
			raise ValueError('momentum must be between 0 and 1')
		self._momentum = e
	
	@property
	def split(self):
		return self._split

	@split.setter
	def	split(self, e):
		if e <= 0 or e > 1:
			raise ValueError('split must be between 0 and 1')
		self._split = e

	@property
	def tol(self):
		return self._tol

	@tol.setter
	def	tol(self, e):
		if e <= 0:
			raise ValueError('tol must be positive')
		self._tol = e
	
	@property
	def regul(self):
		return self._regul

	@regul.setter
	def	regul(self, e):
		if e <= 0:
			raise ValueError('regul must be positive')
		self._regul = e
	
	@property
	def n_iter_to_change(self):
		return self._n_iter_to_change

	@n_iter_to_change.setter
	def	n_iter_to_change(self, e):
		if type(e) is not int or e <= 0:
			raise ValueError('n_iter_to_change must be positive')
		self._n_iter_to_change = e
	
	@property
	def early_stopping(self):
		return self._early_stopping

	@early_stopping.setter
	def	early_stopping(self, e):
		if type(e) is not bool:
			raise ValueError('early_stopping must be a boolean')
		self._early_stopping = e

	@property
	def nesterov(self):
		return self._nesterov

	@nesterov.setter
	def	nesterov(self, e):
		if type(e) is not bool:
			raise ValueError('nesterov must be a boolean')
		self._nesterov = e

	@property
	def distrib_name(self):
		return self._distrib_name

	@distrib_name.setter
	def	distrib_name(self, d):
		if d not in ['normal', 'uniform', 'LCnormal', 'LCuniform', 'XGnormal', 'XGuniform', 'Henormal', 'Heuniform']:
			raise ValueError('distribution must be normal, uniform, LCnormal, LCuniform, XGnormal, XGuniform, Henormal, or Heuniform')
		self._distrib_name = d
	
	@property
	def opti_name(self):
		return self._opti_name

	@opti_name.setter
	def	opti_name(self, o):
		if o not in ['sgd', 'adam']:
			raise ValueError('optimizer must be sgd or adam')
		self._opti_name = o
	
	@property
	def beta1(self):
		return self._beta1

	@beta1.setter
	def	beta1(self, b):
		if b <= 0 or b >= 1:
			raise ValueError('beta1 must be between 0 and 1')
		self._beta1 = b
	
	@property
	def beta2(self):
		return self._beta2

	@beta2.setter
	def	beta2(self, b):
		if b <= 0 or b >= 1:
			raise ValueError('beta2 must be between 0 and 1')
		self._beta2 = b
	
	def	_normalize(self, train_input):
		"""Normalize the input data
		
		Parameters:
			train_input: data frame
				Input data

		Returns:
			train_input: data frame
				Normalized input data
		"""

		self.norm = {
			'mean': [],
			'std': []
		}
		for i in train_input.columns[:]:
			self.norm['mean'].append(train_input[i].mean())
			self.norm['std'].append(train_input[i].std())
			train_input[i] = (train_input[i] - train_input[i].mean()) / train_input[i].std()

		return train_input

	def	_seperate(self, train_input, train_output):
		"""Split the data into train and test sets
		
		Parameters:
			train_input: data frame
				Input data
			train_output: data frame
				Output data

		Returns:
			train_input: data frame
				Training input
			train_output: data frame
				Training output
			test_input: data frame
				Testing input
			test_output: data frame
				Testing output
		"""

		file = pd.concat([train_output, train_input], axis=1, ignore_index=True)
		data_train, data_test = seperate(file, 0, self.split, self.seed)
		return data_train.drop(columns=0), data_train[0], data_test.drop(columns=0), data_test[0]
	
	def	__init_data(self, train_input, train_output):
		"""Initialize the data

		Parameters:
			train_input: data frame
				Input data
			train_output: data frame
				Output data
		"""

		train_input = self._normalize(train_input)
		if self.early_stopping:
			train_input, train_output, test_input, test_output = self._seperate(train_input, train_output)
			if not train_output.size or not test_output.size:
				raise Exception('split invalid, no training or testing data')
		self.input = np.array(train_input)
		self.nb_params = train_input.shape[1]
		self.nb_samples = train_input.shape[0]

		self.layers.insert(0, Layer(self.nb_params))
		self.layers_sizes.insert(0, self.nb_params)

		self.unique, self.train_output = np.unique(train_output, return_inverse=True)
		if self.output_layer_activation == 'sigmoid' and len(self.unique) != 2:
			raise Exception("sigmoid ativation on outer layer can't be used with more than 2 categories")
		if self.output_layer_activation == 'softmax':
			self.size_output = len(self.unique)
			self.output = np.eye(self.size_output)[self.train_output]
		else:
			self.size_output = 1
			self.output = np.array([[o] for o in self.train_output])
		self.layers.append(Layer(self.size_output, self.output_layer_activation))
		self.layers_sizes.append(self.size_output)

		if self.early_stopping:
			self.test_input = test_input
			self.test_output = [np.where(self.unique == o)[0][0] for o in test_output]

	def	__init_weights(self):
		"""Initialize the weights and biases"""

		for i in range (1, len(self.layers_sizes)):
			if self.opti_name == 'adam':
				self.mt_b.append(np.zeros((1, self.layers_sizes[i])))
				self.vt_b.append(np.zeros((1, self.layers_sizes[i])))
				self.mt_w.append(np.zeros((self.layers_sizes[i - 1], self.layers_sizes[i])))
				self.vt_w.append(np.zeros((self.layers_sizes[i - 1], self.layers_sizes[i])))
			weights = np.array(self.distrib(self.rng, self.layers_sizes[i - 1], self.layers_sizes[i]))
			self.weights.append(weights)
			bias = np.array(self.distrib(self.rng, 1, self.layers_sizes[i]))
			self.bias.append(bias)
			self.velocity_w.append(np.zeros(weights.shape))
			self.velocity_b.append(np.zeros(bias.shape))

	def	add_layer(self, size, function):
		"""Add a layer to the model

		Parameters:
			size: int
				Number of neurons in the layer
			function: str
				Activation function of the layer
		"""
	
		self.layers.append(Layer(size, function))
		self.layers_sizes.append(size)
		self.size += 1


	def fit(self, train_input, train_output):
		"""Fit the model to the data

		Parameters:
			train_input: data frame
				Input data
			train_output: data frame
				Output data
		"""
	
		self.__init_data(train_input, train_output)
		self.__init_weights()
		self.__train()

	def __check_early_stopping(self, metrics: Metrics, no_changes, i):
		"""Check if early stopping should be triggered
		
		Parameters:
			metrics: Metrics object
				object storing model's metrics
			no_changes: int
				Number of epochs with no changes
			i: int
				Current epoch
			
		Returns:
			no_changes: int
				Number of epochs with no changes
		"""

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
		"""Train the model using the parameters and data provided."""
		if self.name == None or type(self.name) is not str:
			self.name = self.opti_name + self.output_layer_activation
		self.metrics = Metrics(self.name, self.output_layer_activation)
		steps = 0
		no_changes = 0
		self.batch_size = min(self.batch_size, self.nb_samples)
		for i in range (1, self.epochs + 1):

			batch_set = batch_init(self.input.shape[0])
			self.metrics.loss = 0
			self.metrics.acc = 0
			while (batch_set):

				batch_index = get_batch(batch_set, self.batch_size)
				input = self.input[batch_index]

				self.__feedforward(input)
				self.metrics.get_train_loss_and_acc(self, batch_index)
				steps += 1
				self.__backprop(batch_index, steps)

			self.metrics.add_loss_acc(self.nb_samples)

			if self.early_stopping:
				self.__feedforward(self.test_input)
				self.metrics.get_test_loss_and_acc(self)

			print(
				'epochs: {}/{}'.format(i, self.epochs),
				' - loss: {:.4}'.format(self.metrics.train_loss[-1]),
				' - val_loss: {:.4}'.format(self.metrics.test_loss[-1]) if self.early_stopping else ""
			),

			no_changes = self.__check_early_stopping(self.metrics, no_changes, i)
			if no_changes >= self.n_iter_to_change:
				print('Stopped early at epoch', self.best_epoch if self.early_stopping else i)
				if self.early_stopping:
					self.weights = [w.copy() for w in self.best_weights]
					self.bias = [b.copy() for b in self.best_bias]
					self.__feedforward(self.test_input)
					self.converged_in = self.best_epoch
					self.metrics.get_confusion_and_metrics(self, self.test_output)
				else:
					self.converged_in = i
				break

	def	predict(self, input, output):
		"""Predict the output of the input data
		
		Parameters:
			input: data frame
				Input data
			output: data frame
				Output data
		"""
	
		for (i, mean, std) in zip(input.columns, self.norm['mean'], self.norm['std']):
			input[i] = (input[i] - mean) / std
		if self.converged_in == 0:
			print("Can't predict on an untrained model")
			return
		self.predict_output = output
		self.__feedforward(input)

	def __feedforward(self, input):
		"""Feed the input data through the model

		Parameters:
			input: data frame
				Input data
		"""
	
		self.layers[0].neurons = input
		for i in range (1, self.size):
			self.layers[i].neurons_base = np.dot(self.layers[i - 1].neurons, self.weights[i - 1]) + self.bias[i - 1]
			self.layers[i].neurons = self.layers[i].activation(self.layers[i].neurons_base)
		return self.layers[-1].neurons

	def	__backprop(self, batch_index, steps):
		"""Backpropagate the error through the model

		Parameters:
			batch_index: list
				Indices of the batch
			steps: int
				Number of steps used for adam optimizer
		"""

		self.optimizer(self, batch_index, steps)
		