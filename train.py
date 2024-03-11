#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class MultiLayerPerceptron:
	def __init__(self, data: pd.DataFrame):
		self.result = np.array(data[1])
		data = data.drop(columns = [0, 1])
		self.data = data
		self.size = self.result.size
		self.layers = [len(data.columns), 10, 10, 1]
		self.weights = []
		for i in range(len(self.layers) - 1):
			self.weights.append(np.zeros((self.layers[i], self.layers[i + 1]), float))
	
	def feedforward(self):
		n = random.randrange(0, self.size)
		intermediate = self.data.loc[n]
		print(intermediate)
		for layer in self.weights:
			intermediate = np.dot(intermediate, layer)
			intermediate = sigmoid(intermediate)

		print(intermediate)

data_train = pd.read_csv('data_train.csv', header = None)
mlp = MultiLayerPerceptron(data_train)
mlp.feedforward()