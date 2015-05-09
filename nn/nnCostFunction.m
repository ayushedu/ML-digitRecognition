# Back propogation algo implementation
function [J, gradient] = nnCostFunction(X, Theta1, Theta2, y,num_labels, lambda)

  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  m = size(X,1);

  ## calculate activation units 
  a2 = sigmoid(X*Theta1');
  a2 = [ones(size(a2,1), 1) a2]; # add ones 
  a3 = sigmoid(a2 * Theta2');  
  y_vec = bsxfun(@eq, y, 1:num_labels); # convert y to vector of 0 and 1.  

  ## Back propogation algo
  ### Calculate cost
  p = zeros(m, num_labels);

  for i = 1:m
    for k = 1:num_labels
      p(i,k) = log(a3(i,k))'*y_vec(i,k) + log(1-a3(i,k))' * (1-y_vec(i,k));
    endfor 
  endfor

  J = -1/m * sum(sum(p));

  ### Add regularization
  J = J + (lambda./(2.*m)) * sum( sum( Theta1(:,2:end).^2 ) ) + sum(  sum( Theta2(:,2:end).^2 ) );
	    
### calculate gradient
delta3 = a3 - y_vec;
delta2 = (delta3 * Theta2)  .* a2.*(1-a2);
delta2 = delta2(:,2:end);

Theta2_grad = 1./m * delta3'*a2;
Theta1_grad = 1./m * delta2'*X;

### add regularization to all except first col
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda./m * Theta2(:,2:end));
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda./m * Theta1(:,2:end);

## Unroll gradient
gradient = [Theta1_grad(:); Theta2_grad(:)];
endfunction 