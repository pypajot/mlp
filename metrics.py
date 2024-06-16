import numpy as np
import matplotlib.pyplot as plt

from loss import loss_func

class Metrics:
	def	__init__(self):
		self.acc = 0
		self.loss = 0
		self.train_loss = []
		self.test_loss = []
		self.train_acc = []
		self.test_acc = []
		self.test_precision = 0
		self.test_recall = 0
		self.test_f1score = 0
		self.confusion_matrix = np.empty((0, 0))

	def get_test_loss_and_acc(self, mlp):
		loss = 0
		acc = 0
		mlp.feedforward(mlp.test_input)
		for i in range (mlp.test_output.size):
			loss += loss_func[mlp.output_layer_activation](mlp.layers[-1].neurons[i], mlp.test_output[i])
			# math.log(mlp.layers[-1].neurons[i][mlp.test_output[i]])
			acc += mlp.test_output[i] == np.argmax(mlp.layers[-1].neurons[i])
		loss /= mlp.test_output.size
		acc /= mlp.test_output.size
		# for wei in mlp.weights:
		# 	loss += np.sum(np.power(wei, 2)) * mlp.regul * mlp.learning_rate
		self.test_loss.append(loss)
		self.test_acc.append(acc)

	def get_confusion_and_metrics(self, mlp):
		size = max(mlp.test_output) + 1
		self.confusion_matrix = np.zeros((size, size))
		self.acc = 0
		self.loss = 0
		for i in range (mlp.test_output.size):
			self.loss += loss_func[mlp.output_layer_activation](mlp.layers[-1].neurons[i], mlp.test_output[i])
			self.confusion_matrix[mlp.test_output[i]][np.argmax(mlp.layers[-1].neurons[i])] += 1
			# true_pos += int(mlp.test_output[i] and mlp.test_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
			# true_neg += int((not mlp.test_output[i]) and mlp.test_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
			# false_pos += int(mlp.test_output[i] and mlp.test_output[i] != np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
			# false_neg += int((not mlp.test_output[i]) and mlp.test_output[i] != np.argmax(mlp.layers[mlp.size - 1].neurons[i]))
		# acc = (true_pos + true_neg) / mlp.test_output.size
		if size == 2:
			self.acc = (self.confusion_matrix[0][0] + self.confusion_matrix[1][1]) / np.sum(self.confusion_matrix)
			self.precision = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])
			self.recall = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[0][1])
		else:
			for i in range(size):
				self.acc += self.confusion_matrix[i][i]
				self.precision += self.confusion_matrix[i][i] / (size * (np.sum(self.confusion_matrix[i][:])))
				self.recall += self.confusion_matrix[i][i] / (size * (np.sum(self.confusion_matrix[:][i])))
			self.acc /= np.sum(self.confusion_matrix)
		self.f1score = 2 * self.precision * self.recall / (self.precision + self.recall)


	def get_train_loss_and_acc(self, mlp, batch_index):
		for i in batch_index:
			# print(mlp.output_layer_activation)
			# print(mlp.layers[mlp.size - 1].neurons[i])
			# print(mlp.layers[-1].neurons[batch_index.index(i)])
			self.loss += loss_func[mlp.output_layer_activation](mlp.layers[-1].neurons[batch_index.index(i)], mlp.train_output[i]) / mlp.train_output.size
			# self.loss -= math.log(mlp.layers[-1].neurons[batch_index.index(i)][mlp.train_output[i]]) / mlp.train_output.size
			self.acc += (mlp.train_output[i] == np.argmax(mlp.layers[mlp.size - 1].neurons[batch_index.index(i)])) / mlp.train_output.size
		# self.loss /= mlp.train_output.size
		# self.acc /= mlp.train_output.size
		# for wei in mlp.weights:
		# 	loss += np.sum(np.power(wei, 2)) * mlp.regul * mlp.learning_rate
		# self.train_loss.append(loss)
		# self.train_acc.append(acc)

	def add_loss_acc(self):
		self.train_loss.append(self.loss)
		self.train_acc.append(self.acc)

	def show(self):
		fig, axs = plt.subplots(1, 2)
		axs[0].plot(range(len(self.train_loss)), self.train_loss, label='train')
		axs[0].plot(range(len(self.test_loss)), self.test_loss, label='validation')
		axs[1].plot(range(len(self.train_acc)), self.train_acc, label='train')
		axs[1].plot(range(len(self.test_acc)), self.test_acc, label='validation')
		axs[0].legend()
		axs[1].legend()
		fig.set_size_inches((15, 5))
		text_str = '\n'.join((
			'acc = {:.2}'.format(self.acc),
			'precision = {:.2}'.format(self.precision),
			'recall = {:.2}'.format(self.recall),
			'f1score = {:.2}'.format(self.f1score)
		))
		fig.text(0.92, 0.73, s=text_str)
		plt.show()