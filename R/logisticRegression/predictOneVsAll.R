predictOneVsAll <- function(all_theta, X) {
  ## initialize vars
  m <- dim(X)[1]
  num_labels < dim(all_theta)[1]
  
  p <- matrix(data=0, nrow = m)
  X <- cbind(1, X) # add ones to data
  print('-------')
  print(dim(X))
  print(dim(all_theta))
  h <- sigmoid(X %*% t(all_theta))
  
  return(max.col(h))
  
}