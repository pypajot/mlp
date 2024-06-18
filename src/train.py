#!/usr/bin/env python3

import pandas as pd
import argparse
import pickle
import os
import sklearn.neural_network as sk

import sys
sys.path.append('/mnt/nfs/homes/ppajot/Documents/mlp/src/modules')

from mlp import MultiLayerPerceptron

def main():

	parser = argparse.ArgumentParser(
		prog='train.py',
		description='Train a MLPClassifier on some data'
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
	parser.add_argument('-n', '--nesterov', action='store_true')
	parser.add_argument('-A', '--output_layer_activation', default='softmax')
	parser.add_argument('-d', '--distribution', default='XGuniform')
	parser.add_argument('-m', '--momentum', type=float, default=0.9)
	parser.add_argument('-r', '--regul', type=float, default=0.0001)
	parser.add_argument('-t', '--tol', type=float, default=0.0001)
	parser.add_argument('-i', '--n_iter_to_change', type=int, default=10)
	parser.add_argument('-s', '--seed', type=int, default=None)
	parser.add_argument('-b1', '--beta1', type=float, default=0.9)
	parser.add_argument('-b2', '--beta2', type=float, default=0.999)
	parser.add_argument('-N', '--name', default=None)



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
		test_output = data_test[1]
		test_input = data_test.drop(columns=[0, 1])
	except Exception:
		print('test data is invalid')
		exit(2)

	try:
		model = MultiLayerPerceptron(
			layers_sizes=args.layers,
			optimizer=args.optimizer,
			epochs=args.epochs,
			activation_func=args.activation,
			output_layer_activation=args.output_layer_activation,
			batch_size=args.batch_size,
			learning_rate=args.learning_rate,
			early_stopping=args.early_stopping,
			momentum=args.momentum,
			nesterov=args.nesterov,
			regul=args.regul,
			tol=args.tol,
			n_iter_to_change=args.n_iter_to_change,
			beta1=args.beta1,
			beta2=args.beta2,
			distrib=args.distribution,
			seed=args.seed,
			name=args.name
		)
		model.fit(train_input, train_output, test_input, test_output)
		model.metrics.show()
	except Exception as e:
		print(e)
		exit(1)


	# scikit = sk.MLPClassifier(solver='adam', early_stopping=True)
	# scikit.fit(train_input, train_output)
	# import matplotlib.pyplot as plt
	# plt.plot(scikit.loss_curve_)
	# plt.show()



	try:
		file = open('model.pkl', 'wb')
		pickle.dump(model, file)
	except Exception as e:
		print(e)
		print('Error opening model.pkl')
		exit(2)

	try:
		if os.path.isfile('metrics.pkl'):
			metric_file_read = open('metrics.pkl', 'rb')
			metrics = pickle.load(metric_file_read)
		else:
			metrics = []
		metrics.append(model.metrics)
		metric_file_write = open('metrics.pkl', 'wb')
		metrics = pickle.dump(metrics, metric_file_write)
	except Exception as e:
		print(e)
		print('Error pickling metrics')
		exit(2)

if __name__ == '__main__':
	main()
