import numpy as np

def normal(rng, F_in, F_out):
	limit = 0.4
	return rng.normal(0.0, limit, (F_in, F_out))

def uniform(rng, F_in, F_out):
	limit = 0.2
	return rng.uniform(-limit, limit, (F_in, F_out))

def	LCnormal(rng, F_in, F_out):
	limit = np.sqrt(1 / F_in)
	return rng.normal(0.0, limit, (F_in, F_out))

def LCuniform(rng, F_in, F_out):
	limit = np.sqrt(3 / F_in)
	return rng.uniform(-limit, limit, (F_in, F_out))

def	XGnormal(rng, F_in, F_out):
	limit = np.sqrt(2 / (F_in + F_out))
	return rng.normal(0.0, limit, (F_in, F_out))

def XGuniform(rng, F_in, F_out):
	limit = np.sqrt(6 / (F_in + F_out))
	return rng.uniform(-limit, limit, (F_in, F_out))

def	Henormal(rng, F_in, F_out):
	limit = np.sqrt(2 / F_in)
	return rng.normal(0.0, limit, (F_in, F_out))

def Heuniform(rng, F_in, F_out):
	limit = np.sqrt(6 / F_in)
	return rng.uniform(-limit, limit, (F_in, F_out))

distribution = {
		'normal': normal,
		'uniform': uniform,
		'LCnormal': LCnormal,
		'LCuniform': LCuniform,
		'XGnormal': XGnormal,
		'XGuniform': XGuniform,
		'Henormal': Henormal,
		'Heuniform': Heuniform,

}