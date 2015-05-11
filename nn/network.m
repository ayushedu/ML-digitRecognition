# Neural network to recognize hand written digits

fprintf('Running digit recognition for neural network.\n');

## Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
lambda = 0;

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
a2 = feedfwd(X, Theta1);
a2 = [ones(size(a2,1),1),a2]; # Add ones column
hx = feedfwd(a2, Theta2);

## Train the theta

initial_nn_params = [Theta1(:);Theta2(:)]; # unroll the theta

### maximum no of iterations before optimization stops.
options = optimset('MaxIter',50);

costFunction = @(p) nnCostFunction(X, Theta1, Theta2, y,num_labels, ...
				   lambda);
### find local min
[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

### reshape Theta from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + ...
						  1)), ...
		 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size ...
						      + 1))):end), ...
		 num_labels, (hidden_layer_size + 1)); 
### Predict
fprintf('Predicting.\n');
pred = predict(Theta1, Theta2, X(:,2:end));

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

