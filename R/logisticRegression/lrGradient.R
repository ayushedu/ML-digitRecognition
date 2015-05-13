# logistic regression gradient
lrGradient <- function(theta, X, y, lambda) {
  
  theta <- as.matrix(theta)
  y <- as.matrix(y)
  
  ## initialize vars
  m <- dim(y)[1] # no of training examples
  mask <- matrix(data=1, nrow=dim(theta)[1], ncol=dim(theta)[2])
  grad <- mask * 0
  
  mask[1] <- 0
  
  z <- sigmoid(X %*% theta)

  grad <- (1/m) * t(X) %*% (z-y) + (lambda/m) * theta * mask
  
  grad <- cbind(c(grad))
  
  #print(paste('Returning gradient',dim(grad)[1], dim(grad)[2]))
  
  return(grad)
  
}