#!/usr/bin/env python3

import pandas as pd
import argparse
import numpy as np

from modules.utils import split_df

def seperate(file, output, split, seed=None):
	unique = np.unique(file[output])
	categories = [file[file[output] == u] for u in unique]

	data_train = pd.DataFrame()
	data_validation = pd.DataFrame()
	for c in categories:
		train, validation = split_df(c, split, seed)
		data_train = pd.concat([data_train, train])
		data_validation = pd.concat([data_validation, validation])
	return data_train, data_validation

def main():

	parser = argparse.ArgumentParser(
		prog='seperate.py',
		description='Split data into train and validation files'
	)
	parser.add_argument('data')
	parser.add_argument('-s', '--split', type=float, default=0.8)
	parser.add_argument('-r', '--random-seed', type=int, default=None)
	parser.add_argument('-o', '--output', type=int, default=1)

	args = parser.parse_args()

	split = args.split
	seed = args.random_seed

	if split > 1 or split <= 0:
		print('split must be between 0 and 1')
		exit(1)

	try:
		file = pd.read_csv(args.data, header = None)
	except Exception as e:
		print('Error opening file')
		print(e)
		exit(1)

	try:
		data_train, data_validation = seperate(file, args.output, split, seed)
	except Exception as e:
		print('error during seperation')
		print(e)
	
	try:
		data_train.to_csv('data_train.csv', header = False, index = False)
		data_validation.to_csv('data_validation.csv', header = False, index = False)
	except Exception as e:
		print('error saving sets to csv')
		print(e)
	
if __name__ == '__main__':
	main()