function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
yv = bsxfun(@eq, y, 1:num_labels);
a2 = sigmoid(X*Theta1');

% Add ones to the a1 data matrix
a2 = [ones(size(a2,1), 1) a2];
a3 = sigmoid(a2*Theta2');

p = zeros(m,num_labels);
for i = 1:m
   for k = 1:num_labels
       p(i,k) = -1 * ( log(a3(i,k))'*yv(i,k) + log(1-a3(i,k))'*(1-yv(i,k)) );
   end
end

J = 1./m * sum(sum(p));

ThetaSub1 = Theta1(:,2:end).^2;
ThetaSub2 = Theta2(:,2:end).^2;

J =  J + lambda./(2*m) * (sum(sum(ThetaSub1)) + sum(sum(ThetaSub2)));

% -------------------------------------------------------------
% Back propogation algo
delta3 = a3 - yv;

delta2 = (delta3 * Theta2)  .* a2.*(1-a2);
delta2 = delta2(:,2:end);

Theta2_grad = 1./m * delta3'*a2;
Theta1_grad = 1./m * delta2'*X;

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda./m * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda./m * Theta1(:,2:end);



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
