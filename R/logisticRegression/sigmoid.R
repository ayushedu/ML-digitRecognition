sigmoid <- function(z) {
  h <- 1/(1+exp(-z));
  
  return(h)
}