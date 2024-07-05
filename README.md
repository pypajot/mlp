# Multilayer perceptron
## Description
This is a school project for 42 Paris. The goal of this project is to train a multilayer perceptron to be able to classify tissue sample taken from tumors. The class in itself is able to be trained to classify various data.
## Usage
make or make install: install the libraries located in requirements.txt
make mclean: delete the metrics.pkl file
make fclean: delete all pickle files and generated csv

seperate.py takes as argument a csv file with and given an output column, sepearates the files into training and validation sets.

train.py takes a csv file as argument. You can choose which columns to ignore and which is the output. It then trains a model on the data. The model is saved in model.pkl, and loss history is saved in metrics.pkl.

predict.py takes a validation sets as argument, loads model.pkl and evaluates the model using differents metrics.

compare.py compares the convergence rates of differents models using metrics.pkl

