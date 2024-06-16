import math

def categorical_loss(neuron, category):
	return -math.log(neuron[category])

def	binary_loss(neuron, category):
	return -(category * math.log(neuron) + (1 - category) * math.log(1 - neuron))

loss_func = {
	'softmax': categorical_loss,
	'sigmoid': binary_loss
}