#!/usr/bin/env python3

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('/mnt/nfs/homes/ppajot/Documents/mlp/src/modules')

def	main():

	parser = argparse.ArgumentParser(
		prog='predict.py',
		description='Predict result on a given set of data from a already trained model'
	)
	parser.add_argument('data')
	args = parser.parse_args()

	try:
		data = pd.read_csv(args.data, header = None)
		data[data == 'B'] = 0
		data[data == 'M'] = 1
		output = data[1]
		input = data.drop(columns=[0, 1])
	except Exception:
		print('test data is invalid')
		exit(2)

	try:
		data_file = open('model.pkl', 'rb')
		model = pickle.load(data_file)
	except Exception as e:
		print('Error opening model.pkl')
		print(e)
		exit(1)

	# try:
	# 	norm_file = open('norm.pkl', 'rb')
	# 	norm = pickle.load(norm_file)
	# except Exception as e:
	# 	print('Error opening norm.pkl')
	# 	print(e)
	# 	exit(1)

	# try:
	# 	for i in input.columns:
	# 		input[i] = (input[i] - norm['mean'][i - 2]) / norm['std'][i - 2]
	# except Exception as e:
	# 	print('Error normalizing data')
	# 	print(e)
	# 	exit(1)
	
	# try:
	model.predict(input, output)
	model.metrics.get_confusion_and_metrics(model, model.predict_output)
	fig, axs = plt.subplots(1, 2)
	model.metrics.show_confusion_and_metrics(axs[0], axs[1])
	plt.show()
	# except Exception as e:
	# 	print('Error during prediction')
	# 	print(e)
	# 	exit(2)


if __name__ == '__main__':
	main()