import numpy as np

def	split_df(df, split, seed):	
	"""Split a data frame into train and test sets

	Parameters:
		df: data frame
			Data frame to split
		split: float
			Percentage of data to use for training
		seed: int
			Random seed for shuffling data

	Returns:
		train_df: data frame
			Train set
		test_df: data frame
			Test set
	"""

	train_df = df.sample(None, split, random_state=seed)
	test_df = df.drop(train_df.index)
	return train_df, test_df

def batch_init(size):
	"""Initialize a batch of indices

	Parameters:
		size: int
			Size of the batch set

	Returns:
		set: list
			Shiffled list of indices
	"""

	set = list(range(size))
	np.random.shuffle(set)
	return set

def	get_batch(set, size):
	"""Get a batch of indices

	Parameters:
		set: list
			List of indices
		size: int
			Size of the batch

	Returns:
		batch: list
			Batch of indices
	"""

	size = min(size, len(set))
	batch = set[:size]
	del set[:size]
	return batch
