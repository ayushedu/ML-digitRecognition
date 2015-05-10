# Digit Recognition code using neural network
setwd('/Users/snowleopard/Documents/workspace/kaggle/digit-recognition/R/');
print('Running digit recognition for neural network')

## import functions
source('sigmoid.R')
source('costFunction.R')

## initialize paramters
input_layer_size = 784; # 28x28 input images
hidden_layer_size = 25
num_labels = 10
lambda = 0

## Load data
print("Loading data")
data = read.csv('../data/train.csv')
class(data)
y = data[,1]
x = data[,2:dim(data)[2]]
x = cbind(1, x) # add ones column
rm(data) # remove original data object from memory

## Randomly initialize theta between 0 and 1
print('Initializing theta...')
Theta1 = replicate(input_layer_size + 1, rnorm(hidden_layer_size, mean = 0, sd = 1))
Theta2 = replicate(hidden_layer_size + 1, rnorm(num_labels, mean = 0, sd = 1))

## Feed forward algo
print('Runnning feedfwd...')
a2 = sigmoid(x*t(Theta1))
a2 = cbind(1, a2) # add ones col
hx = sigmoid(a2*t(Theta2))

## Train the theta
print('Running costFunction...')
initial_nn_params = t(c(Theta1, Theta2))
costFunction(x, Theta1, Theta2, a2, hx, y, num_labels, lambda)
