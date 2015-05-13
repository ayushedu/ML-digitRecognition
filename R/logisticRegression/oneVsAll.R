# Training multiple logistic regression classifier
oneVsAll <- function(X, y, num_labels, lambda) {
  ## initialize vars
  m <- dim(X)[1]
  n <- dim(X)[2]
  all_theta <- matrix(data=0, nrow=num_labels, ncol=n+1)
  
  ## Add ones to data matrix
  X <- cbind(1, X)
  
  for(i in 1:num_labels) {
    theta <- as.matrix(all_theta[i,])
   
    res <- optim(par = theta, fn = lrCostFunction, gr = lrGradient, X, (y==i) * 1, lambda, method = 'CG')
    all_theta[i,] <- res$par
    
    print(paste('Iteration ',i,' cost ', res$value, 'convergence ', res$convergence))
  }
  
  return(all_theta)
  
}