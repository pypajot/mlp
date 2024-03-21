#!/usr/bin/env python3

import pandas as pd
import argparse

def	split_df(df: pd.DataFrame, split):
	train_df = df.sample(None, split)
	test_df = df.drop(train_df.index)
	return train_df, test_df


parser = argparse.ArgumentParser('Normalizes and split data into train and test files')
parser.add_argument('-s', '--split', type=float, default=0.8)
parser.add_argument('-r', '--random-seed', type=int, default=None)

args = parser.parse_args()

split = args.split
seed = args.random_seed

file = pd.read_csv('data.csv', header = None)

file[file == 'B'] = 0
file[file == 'M'] = 1

for i in file.columns[2:]:
	file[i] = (file[i] - file[i].mean()) / file[i].std()

negatives = file[file[1] == 0]
positives = file[file[1] == 1]

negatives_train, negatives_test = split_df(negatives, split)
positives_train, positives_test = split_df(positives, split)

data_train = pd.concat([negatives_train, positives_train])
data_test = pd.concat([negatives_test, positives_test])

data_train.to_csv('data_train.csv', header = False, index = False)
data_test.to_csv('data_test.csv', header = False, index = False)
