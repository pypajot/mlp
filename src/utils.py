import numpy as np
import pandas as pd

def	split_df(df: pd.DataFrame, split, seed):
	train_df = df.sample(None, split, random_state=seed)
	test_df = df.drop(train_df.index)
	return train_df, test_df

def batch_init(size):
	set = list(range(size))
	np.random.shuffle(set)
	return set

def	get_batch(set, size):
	size = min(size, len(set))
	batch = set[:size]
	del set[:size]
	return batch
