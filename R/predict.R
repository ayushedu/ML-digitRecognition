predict <- function() {
  p = matrix(data = 0, nrow = dim(x)[1], ncol = 1)
  
  h1 = sigmoid(x %*% t(Theta1))
  
  ## add one to h
  h1 = as.matrix(cbind(1, h1))
  h2 = sigmoid(h1 %*% t(Theta2))
  
  #p = apply(h2, 1, max)
  p = max.col(h2)
  return(p)
}