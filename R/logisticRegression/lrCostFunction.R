# Logistic regression cost function
lrCostFunction <- function(theta, X, y, lambda) {
  
  # Initialize vars
  m <- length(y) # no of training examples
  J <- 0
  
  #print(paste('x ',class(X),' theta ', class(theta)))
  
  z <- sigmoid(X %*% theta)
  
  #print(paste('theta ', dim(theta)))
  
  # calculate cost
  J <- (-1/m) * sum( t(log(z)) %*% y + t(log(1-z)) %*% (1-y) )
  J <- J + (lambda / (2*m)) * sum(t(theta) %*% theta - theta[1]^2) # regularize
  
  #print(paste('J ', J))
  
  return(J)
    
}