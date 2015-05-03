%% Kaggle competition - Digit Recognition

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% ==== Plot data ====
fprintf('Running digit recognition ... \n');

%% ==== Loading and visualizing data ====

% Load training data
fprintf('Loading and Visualizing Data ...\n')
% fid = fopen('../data/train.csv');
% out = textscan(fid,'%s%f%f','delimiter',',');
% fclose(fid);
% fprintf(summary(out));
data = csvread('../data/train.csv',1);
y = data(:,1);
X = data(:,2:end);
clear data;     % Remove data variable

% Randomly select 100 data points
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==== Loading parameters ====

% Initialize theta parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% ==== Train NN ====
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ==== Predict ====
pred = predict(initial_Theta1, initial_Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);