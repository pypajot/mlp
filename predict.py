#!/usr/bin/env python3

import pickle
import pandas as pd
import matplotlib.pyplot as plt

def	main():
	# data = pd.read_csv('data.csv')


	try:
		data = pd.read_csv('data.csv', header = None)
		data[data == 'B'] = 0
		data[data == 'M'] = 1
		output = data[1]
		input = data.drop(columns=[0, 1])
	except Exception:
		print('test data is invalid')
		exit(2)

	data_file = open('model.pkl', 'rb')
	model = pickle.load(data_file)

	norm_file = open('norm.pkl', 'rb')
	norm = pickle.load(norm_file)

	for i in input.columns:
		input[i] = (input[i] - norm['mean'][i - 2]) / norm['std'][i - 2]

	model.predict(input, output)
	model.metrics.get_confusion_and_metrics(model, model.predict_output)
	fig, axs = plt.subplots(1, 2)
	model.metrics.show_confusion_and_metrics(axs[0], axs[1])
	plt.show()


if __name__ == '__main__':
	main()