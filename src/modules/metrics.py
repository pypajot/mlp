import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from modules.loss import loss_func

class Metrics:
	def	__init__(self, name, activation):
		self.acc = 0
		self.loss = 0
		self.loss_func = loss_func[activation]
		self.train_loss = []
		self.test_loss = []
		self.train_acc = []
		self.test_acc = []
		self.test_precision = 0
		self.test_recall = 0
		self.test_f1score = 0
		self.confusion_matrix = np.empty((0, 0))
		self.name = name

	def get_test_loss_and_acc(self, mlp):
		loss = 0
		acc = 0
		
		for i in range (mlp.layers[-1].neurons.shape[0]):
			# print(mlp.layers[-1].neurons[i], mlp.test_output[i])
			loss += self.loss_func(mlp.layers[-1].neurons[i], mlp.test_output[i])
			if mlp.output_layer_activation == 'softmax':
				acc += mlp.test_output[i] == np.argmax(mlp.layers[-1].neurons[i])
			else:
				acc += mlp.test_output[i] == round(mlp.layers[-1].neurons[i][0])
		loss /= mlp.layers[-1].neurons.shape[0]
		acc /= mlp.layers[-1].neurons.shape[0]

		self.test_loss.append(loss)
		self.test_acc.append(acc)

	def get_confusion_and_metrics(self, mlp, output):
		size = max(output) + 1
		self.converged_in = mlp.converged_in
		self.confusion_matrix = np.zeros((size, size))
		self.acc = 0
		self.loss = 0
		for i in range (mlp.layers[-1].neurons.shape[0]):
			self.loss += self.loss_func(mlp.layers[-1].neurons[i], output[i])
			if mlp.output_layer_activation == 'softmax':
				self.confusion_matrix[output[i]][np.argmax(mlp.layers[-1].neurons[i])] += 1
			else:
				self.confusion_matrix[output[i]][round(mlp.layers[-1].neurons[i][0])] += 1
		self.loss /= mlp.layers[-1].neurons.shape[0]
		self.test_acc = self.test_acc[:self.converged_in]
		self.test_loss = self.test_loss[:self.converged_in]
		self.train_acc = self.train_acc[:self.converged_in]
		self.train_loss = self.train_loss[:self.converged_in]
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
			self.loss += self.loss_func(mlp.layers[-1].neurons[batch_index.index(i)], mlp.train_output[i])
			if mlp.output_layer_activation == 'softmax':
				self.acc += (mlp.train_output[i] == np.argmax(mlp.layers[-1].neurons[batch_index.index(i)]))
			else:
				self.acc += (mlp.train_output[i] == round(mlp.layers[-1].neurons[batch_index.index(i)][0]))

	def add_loss_acc(self, out_size):
		self.train_loss.append(self.loss / out_size)
		self.train_acc.append(self.acc / out_size)

	def	show_confusion_and_metrics(self, plot_1, plot_2):
		
		plot_1.matshow(self.confusion_matrix, cmap=plt.cm.Blues)
		for i in range(self.confusion_matrix.shape[0]):
			for j in range(self.confusion_matrix.shape[0]):
				plot_1.text(i, j, int(self.confusion_matrix[j, i]), va='center', ha='center', size=15)

		plot_1.set_xlabel('expected')
		plot_1.set_ylabel('prediction')
		plot_1.set_xticks([0, 1], ['False', 'True'])
		plot_1.xaxis.tick_bottom()
		plot_1.set_yticks([0, 1], ['False', 'True'])
		plot_1.set_title('Confusion matrix')

		plot_2.set_axis_off()
		text_str = '\n'.join((
			'loss = {:.3}'.format(self.loss),
			'accuracy = {:.3}'.format(self.acc),
			'precision = {:.3}'.format(self.precision),
			'recall = {:.3}'.format(self.recall),
			'f1 score = {:.3}'.format(self.f1score)
		))
		plot_2.text(0.14, 0.32, s=text_str, size=16)
		plot_2.set_title('Evaluation metrics')

	def show(self, early_stopping):
		fig, axs = plt.subplots(2, 2)
		axs[0][0].plot(range(1, 1 + len(self.train_loss)), self.train_loss, label='train')
		axs[0][0].plot(range(1, 1 + len(self.test_loss)), self.test_loss, label='validation')
		axs[0][0].set_xlabel('epochs')
		axs[0][0].set_ylabel('loss')
		axs[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))
		axs[0][0].legend()
		axs[0][0].set_title('Loss')

		axs[0][1].plot(range(1, len(self.train_acc) + 1), self.train_acc, label='train')
		axs[0][1].plot(range(1, len(self.test_acc) + 1), self.test_acc, label='validation')
		axs[0][1].set_xlabel('epochs')
		axs[0][1].set_ylabel('accurary')
		axs[0][1].xaxis.set_major_locator(MaxNLocator(integer=True))
		axs[0][1].legend()
		axs[0][1].set_title('accuracy')

		if early_stopping:
			self.show_confusion_and_metrics(axs[1][0], axs[1][1])
		else:
			axs[1][0].set_axis_off()
			axs[1][1].set_axis_off()
		
		fig.set_size_inches((10, 9))
		plt.show()