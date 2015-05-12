#Cost Function
cost <- function(Theta1, hidden_layer_size, input_layer_size, 
                 num_labels, hx, yvec) {
  
  ## convert initial_theta to matrix
  theta1 = matrix(theta, nrow = hidden_layer_size, ncol = (input_layer_size + 1))
  theta2 = matrix(theta, nrow = num_labels, ncol = (hidden_layer_size + 1))
  
  ## cost calucation
  p = matrix(data = 0, nrow = m, ncol = num_labels)
  
  for (i in 1:m) {
    for(k in 1:num_labels) {
      p[i,k] = (t(log(hx[i,k])) %*% yvec[i,k]) + 
        (t(log(1 - hx[i,k])) %*% (1 - yvec[1,k]))
    }
  }
  
  J = (-1/m) * sum(p)
  
  ## Add regularization to cost
  J = J + (lambda / (2*m)) * (sum(theta1[2:dim(theta1)[2]] ^ 2) + 
                                sum(theta2[2:dim(theta2)[2]] ^ 2))

#   initial_theta = gradient(theta, hidden_layer_size,input_layer_size, 
#                    num_labels, hx, yvec)
  print(paste('cost ',J, ' theta[1]', theta1[2]))

  return(J)
}