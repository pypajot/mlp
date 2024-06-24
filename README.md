# Multilayer perceptron
## Description
This is a school project for 42 Paris. The goal of this project is to train a multilayer perceptron to be able to classify tissue sample taken from tumors. The class in itself is able to be trained to classify various data.
## Usage
seperate.py takes as argument a csv file with with tissue samples data and seoearates the files into training and validation sets.

train.py takes a csv file as argument. Column 0 is ignored and column 1 must be categories and trains a model on the data. The model is saved in model.pkl, and loss history is saved in metrics.pkl.

predict.py takes a validation sets as argument, loads model.pkl and evaluates the model using differents metrics.

compare.py compares the convergence rates of differents models using metrics.pkl

