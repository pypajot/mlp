#!/usr/bin/env python3

import pandas as pd
import argparse
import pickle

from mlp import MultiLayerPerceptron

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
	parser.add_argument('-A', '--output_layer_activation', default='softmax')

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

	# mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'relu')
	# # mlp.add_layer(100, 'relu')
	# mlp.add_layer(100, 'sigmoid')
	# mlp.add_layer(100, 'sigmoid')


	model.fit(train_input, train_output, test_input, test_output)
	model.metrics.show()

	file = open('model.pkl', 'wb')

	pickle.dump(model, file)

if __name__ == '__main__':
	main()
