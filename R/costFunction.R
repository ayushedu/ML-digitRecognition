# Back propogation algo
costFunction <- function(theta) {
  dim_Theta1 = dim(Theta1)
  dim_Theta2 = dim(Theta2)
  J = 0
  Theta1_grad = matrix(data = 0, nrow = dim_Theta1[1], ncol = dim_Theta1[2])
  Theta2_grad = matrix(data = 0, nrow = dim_Theta2[1], ncol = dim_Theta2[2])
  m = dim(x)[1];
  
  ## convert y lables to binary vector
  yvec = 1 * outer(y, 0:(num_labels-1), FUN = "==")
  
  ## calculate cost
  p = matrix(data = 0, nrow = m, ncol = num_labels)
  
  for (i in 1:m) {
    for(k in 1:num_labels) {
      p[i,k] = (t(log(hx[i,k])) %*% yvec[i,k]) + (t(log(1 - hx[i,k])) %*% (1 - yvec[1,k]))
    }
  }
  
  J = (-1/m) * sum(p)
    
  ## Add regularization to cost
  J = J + (lambda / (2*m)) * (sum(Theta1[2:dim_Theta1[2]] ^ 2) + sum(Theta2[2:dim_Theta2[2]] ^ 2))
  
  ## Calcualte Gradient
  delta3 = hx - yvec
  delta2 = (delta3 %*% Theta2) * a2 * (1-a2)
  delta2 = delta2[,2:dim(delta2)[2]]
  
  Theta2_grad = 1/m * (t(delta3) %*% a2)
  Theta1_grad = 1/m * (t(delta2) %*% x)
  
  ### Regualize theta, except first col
  Theta2_grad[,2:dim_Theta2[2]] = Theta2_grad[,2:dim_Theta2[2]] + lambda/m * Theta2[,2:dim_Theta2[2]]
  Theta1_grad[,2:dim_Theta1[2]] = Theta1_grad[,2:dim_Theta1[2]] + lambda/m * Theta1[,2:dim_Theta1[2]]
  
  ### unroll gradients to vector
  gradient = t(c(Theta1_grad, Theta2_grad))
  
  return(J)
}