#!/usr/bin/env python3

import pandas as pd
import math
import argparse

parser = argparse.ArgumentParser('Normalizes and split data into train and test files')
parser.add_argument('-s', '--split', type=float, default=0.8)
parser.add_argument('-r', '--random-seed', type=int, default=None)

args = parser.parse_args()

split = args.split
seed = args.random_seed

file = pd.read_csv('data.csv', header = None)

size = file[0].size
train_size = math.floor(size * split)
test_size = size - train_size

file = file.sample(None, 1, random_state=seed)

file[file == 'B'] = 0
file[file == 'M'] = 1
for i in range (2, 32):
	file[i] = (file[i] - file[i].min()) / (file[i].max() - file[i].min())
	file[i] = (file[i] - file[i].mean()) / file[i].std()


data_train = file.head(train_size)
data_test = file.tail(test_size)

data_train.to_csv('data_train.csv', header = False, index = False)
data_test.to_csv('data_test.csv', header = False, index = False)
