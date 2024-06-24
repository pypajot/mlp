import numpy as np 

def get_part_deriv(mlp, delta, batch_index, i):
	if (i == mlp.size - 1):
		delta = mlp.layers[i].neurons - mlp.output[batch_index]
	else:
		delta = mlp.layers[i].deriv_acti(mlp.layers[i].neurons) * np.dot(delta, mlp.weights[i].T)
	return delta, np.mean(delta, 0), np.dot(mlp.layers[i - 1].neurons.T, delta) / len(batch_index)

def adam(mlp, batch_index, steps):
	delta = 0
	for i in reversed(range(1, mlp.size)):
		delta, d_bias, d_weights = get_part_deriv(mlp, delta, batch_index, i)
		
		d_bias += mlp.regul * mlp.bias[i - 1][0]
		d_weights += mlp.regul * mlp.weights[i - 1]

		mlp.mt_b[i - 1] = mlp.beta1 * mlp.mt_b[i - 1] + (1 - mlp.beta1) * d_bias
		mlp.mt_w[i - 1] = mlp.beta1 * mlp.mt_w[i - 1] + (1 - mlp.beta1) * d_weights

		mlp.vt_b[i - 1] = mlp.beta2 * mlp.vt_b[i - 1] + (1 - mlp.beta2) * np.power(d_bias, 2)
		mlp.vt_w[i - 1] = mlp.beta2 * mlp.vt_w[i - 1] + (1 - mlp.beta2) * np.power(d_weights, 2)

		mt_corr_b = np.divide(mlp.mt_b[i - 1], 1 - np.power(mlp.beta1, steps))
		mt_corr_w = np.divide(mlp.mt_w[i - 1], 1 - np.power(mlp.beta1, steps))

		vt_corr_b = np.divide(mlp.vt_b[i - 1], 1 - np.power(mlp.beta2, steps))
		vt_corr_w = np.divide(mlp.vt_w[i - 1], 1 - np.power(mlp.beta2, steps))

		mt_corr_b = np.divide(mt_corr_b, np.sqrt(vt_corr_b) + mlp.epsilon)
		mt_corr_w = np.divide(mt_corr_w, np.sqrt(vt_corr_w) + mlp.epsilon)
		mlp.bias[i - 1] -= mlp.learning_rate * mt_corr_b
		mlp.weights[i - 1] -= mlp.learning_rate * mt_corr_w

def gradient_descent(mlp, batch_index, steps):
	delta = 0
	for i in reversed(range(1, mlp.size)):
		delta, d_bias, d_weights = get_part_deriv(mlp, delta, batch_index, i)
	
		d_bias += mlp.regul * mlp.bias[i - 1][0]
		d_weights += mlp.regul * mlp.weights[i - 1]
	
		mlp.velocity_w[i - 1] = mlp.momentum * mlp.velocity_w[i - 1] + mlp.learning_rate * d_weights
		mlp.velocity_b[i - 1] = mlp.momentum * mlp.velocity_b[i - 1] + mlp.learning_rate * d_bias


		if mlp.nesterov:
			mlp.weights[i - 1] -= mlp.momentum * mlp.velocity_w[i - 1] + mlp.learning_rate * d_weights
			mlp.bias[i - 1] -= mlp.momentum * mlp.velocity_b[i - 1] + mlp.learning_rate * d_bias
		else:
			mlp.weights[i - 1] -= mlp.velocity_w[i - 1]
			mlp.bias[i - 1] -= mlp.velocity_b[i - 1]


optimizers = {
	'sgd': gradient_descent,
	'adam': adam
}