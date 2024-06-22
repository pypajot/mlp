#!/usr/bin/env python3

import pickle
import matplotlib.pyplot as plt

import sys
# sys.path.append('/mnt/nfs/homes/ppajot/Documents/mlp/src/modules')
sys.path.append('/home/pierre-yves/Documents/Projects/mlp/src/modules')

def	main():

	try:
		metric_file = open('metrics.pkl', 'rb')
		metrics = pickle.load(metric_file)
	except Exception as e:
		print('Error opening metrics.pkl')
		print(e)
		exit(1)

	fig, axs = plt.subplots(1, 4)

	try:
		for metric in metrics:
			axs[0].plot(metric.train_loss, label=metric.name)
			axs[1].plot(metric.test_loss, label=metric.name)
			axs[2].plot(metric.train_acc, label=metric.name)
			axs[3].plot(metric.test_acc, label=metric.name)
	except Exception as e:
		print('Error accessing metrics data')
		print(e)
		exit(2)
	
	axs[0].legend()
	axs[1].legend()
	axs[2].legend()
	axs[3].legend()
	axs[0].set_title('Train loss')
	axs[1].set_title('Test loss')
	axs[2].set_title('Train accuracy')
	axs[3].set_title('Test accuracy')
	fig.set_size_inches((20, 5))
	plt.show()

if __name__ == '__main__':
	main()