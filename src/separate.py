#!/usr/bin/env python3

import pandas as pd
import argparse

from modules.utils import split_df

def separate(file, split, seed):
	negatives = file[file[1] == 'B']
	positives = file[file[1] == 'M']

	negatives_train, negatives_test = split_df(negatives, split, seed)
	positives_train, positives_test = split_df(positives, split, seed)

	data_train = pd.concat([negatives_train, positives_train])
	data_validation = pd.concat([negatives_test, positives_test])

	return data_train, data_validation

def main():

	parser = argparse.ArgumentParser('Normalizes and split data into train and test files')
	parser.add_argument('data')
	parser.add_argument('-s', '--split', type=float, default=0.8)
	parser.add_argument('-r', '--random-seed', type=int, default=None)

	args = parser.parse_args()

	split = args.split
	seed = args.random_seed

	try:
		file = pd.read_csv(args.data, header = None)
	except Exception as e:
		print('Error opening file')
		print(e)
		exit(1)

	data_train, data_validation = separate(file, split, seed)

	data_train.to_csv('data_train.csv', header = False, index = False)
	data_validation.to_csv('data_validation.csv', header = False, index = False)

if __name__ == '__main__':
	main()