import math

def categorical_loss(neuron, category):
	return -math.log(neuron[category] + 1e-20)

def	binary_loss(neuron, category):
	return -(category * math.log(neuron + 1e-20) + (1 - category) * math.log(1 - neuron + 1e-20))

loss_func = {
	'softmax': categorical_loss,
	'sigmoid': binary_loss
}