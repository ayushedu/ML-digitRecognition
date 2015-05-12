gradient <- function(theta, hidden_layer_size, input_layer_size, 
                     num_labels, hx, yvec) {

  ## convert initial_theta to matrix
  theta1 = matrix(theta, nrow = hidden_layer_size, ncol = (input_layer_size + 1))
  theta2 = matrix(theta, nrow = num_labels, ncol = (hidden_layer_size + 1))
  
  ## Calcualte Gradient
  delta3 = hx - yvec
  delta2 = (delta3 %*% theta2) * a2 * (1-a2)
  delta2 = delta2[,2:dim(delta2)[2]]
  
  Theta2_grad = 1/dim(x)[1] * (t(delta3) %*% a2)
  Theta1_grad = 1/dim(x)[1] * (t(delta2) %*% x)
  
  ## Regualize theta, except first col
  Theta2_grad[,2:dim(theta2)[2]] = Theta2_grad[,2:dim(theta2)[2]] + 
    lambda/m * Theta2[,2:dim(theta2)[2]]
  Theta1_grad[,2:dim(theta1)[2]] = Theta1_grad[,2:dim(theta1)[2]] + 
    lambda/m * Theta1[,2:dim(theta1)[2]]
  
  ### unroll gradients to vector
  gradient = t(c(Theta1_grad, Theta2_grad))
  
  return(gradient)
}