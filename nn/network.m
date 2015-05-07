# Neural network to recognize hand written digits

fprintf('Running digit recognition for neural network.\n');

## Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 35;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

## Load data
fprintf('Loading and Visualizing Data ...\n')
data = csvread('../data/train.csv');
y = data(:,1);
X = data(:,2:end);
X = [ones(size(X,1),1),X]; # Add ones column 
clear data;     % Remove data variable

## Randomly initialize theta
Theta1 = stdnormal_rnd(hidden_layer_size, input_layer_size + 1);
Theta2 = stdnormal_rnd(num_labels, hidden_layer_size + 1);

## Implement sigmoid

## Implement feedfwd
a1 = feedfwd(X, Theta1);
a1 = [ones(size(a1,1),1),a1]; # Add ones column

a2 = feedfwd(a1, Theta2);

## Train the nn
