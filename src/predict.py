#!/usr/bin/env python3

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def	main():

	parser = argparse.ArgumentParser(
		prog='predict.py',
		description='Predict result on a given set of data from a already trained model'
	)
	parser.add_argument('data')
	parser.add_argument('-d', '--drop', nargs='+', type=int, default=[])
	parser.add_argument('-o', '--output', type=int, default=1)
	args = parser.parse_args()

	try:
		data = pd.read_csv(args.data, header = None)
		data = data.drop(columns=args.drop)
		output_literal = data[args.output]
		input = data.drop(columns=[args.output])
	except Exception:
		print('test data is invalid')
		exit(2)

	try:
		data_file = open('model.pkl', 'rb')
		model = pickle.load(data_file)
		output = [np.where(model.unique == o)[0][0] for o in output_literal]
	except Exception as e:
		print('Error opening model.pkl')
		print(e)
		exit(1)
	
	try:
		model.predict(input, output)
		model.metrics.get_confusion_and_metrics(model, model.predict_output)
		fig, axs = plt.subplots(1, 2)
		model.metrics.show_confusion_and_metrics(axs[0], axs[1])
		plt.show()
	except Exception as e:
		print('Error during prediction')
		print(e)
		exit(2)


if __name__ == '__main__':
	main()