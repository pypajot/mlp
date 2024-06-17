#!/usr/bin/env python3

import pandas as pd
import pickle
import argparse

import sys
sys.path.append('modules')

from utils import split_df

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

	file[file == 'B'] = 0
	file[file == 'M'] = 1

	norm = {
		'mean': [],
		'std': []
	}

	try:
		for i in file.columns[2:]:
			norm['mean'].append(file[i].mean())
			norm['std'].append(file[i].std())
			file[i] = (file[i] - file[i].mean()) / file[i].std()
	except Exception as e:
		print('Error normalizing data')
		print(e)
		exit(2)

	try:
		norm_file = open('norm.pkl', 'wb')
		pickle.dump(norm, norm_file)
	except Exception as e:
		print('Error opening norm.pkl')
		print(e)
		exit(1)

	negatives = file[file[1] == 0]
	positives = file[file[1] == 1]

	negatives_train, negatives_test = split_df(negatives, split, seed)
	positives_train, positives_test = split_df(positives, split, seed)

	data_train = pd.concat([negatives_train, positives_train])
	data_test = pd.concat([negatives_test, positives_test])

	data_train.to_csv('data_train.csv', header = False, index = False)
	data_test.to_csv('data_test.csv', header = False, index = False)

if __name__ == '__main__':
	main()