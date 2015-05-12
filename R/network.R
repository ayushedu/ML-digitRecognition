# Digit Recognition code using neural network
setwd('/Users/snowleopard/Documents/workspace/kaggle/digit-recognition/R/');
print('Running digit recognition for neural network')

closeAllConnections()
#rm(list=ls())

## import functions
source('sigmoid.R')
#source('costFunction.R')
source('cost.R')
source('gradient.R')
source('predict.R')
## initialize paramters
input_layer_size = 784; # 28x28 input images
hidden_layer_size = 25
num_labels = 10
lambda = 1

## Load datadi
print("Loading data")
# data = read.csv('../data/train.csv')
# y = data[,1]
# x = data[,2:dim(data)[2]]
# x = as.matrix(cbind(1, x)) # add ones column
# rm(data) # remove original data object from memory
m = dim(x)[1]

## Randomly initialize theta between 0 and 1
print('Initializing theta...')
Theta1 = replicate(input_layer_size + 1, rnorm(hidden_layer_size, mean = 0, sd = 1))
Theta2 = replicate(hidden_layer_size + 1, rnorm(num_labels, mean = 0, sd = 1))

# Theta1 = matrix(data=0, ncol=input_layer_size + 1, nrow=hidden_layer_size)
# Theta2 = matrix(data=0, ncol=hidden_layer_size + 1, nrow=num_labels)

## Feed forward algo
print('Runnning feedfwd...')
a2 = sigmoid(x %*% t(Theta1))
a2 = cbind(1, a2) # add ones col
hx = sigmoid(a2 %*% t(Theta2))

## convert y lables to binary vector
yvec = 1 * outer(y, 0:(num_labels-1), FUN = "==")

## Train the theta
print('Running costFunction...')
initial_theta = t(c(Theta1, Theta2))

#costFunction()
#  res <- optim(par = initial_theta, fn = cost, gr = gradient, method = "CG", 
#               control = list(maxit=50))
#library(optimx)
# res <- optim(par = initial_theta, fn = cost,control = list(trace=1),
#               method = c("CG"),hidden_layer_size=hidden_layer_size, 
#               input_layer_size=input_layer_size, 
#               num_labels=num_labels, hx=hx, yvec=yvec)



## - --------------
for(i in 1:4) {
  theta = gradient(initial_theta, hidden_layer_size, input_layer_size, num_labels, hx, yvec)
  J = cost(theta, hidden_layer_size, input_layer_size, num_labels, hx, yvec);
  
}
## reshape theta from par
Theta1 = matrix(theta, nrow = hidden_layer_size, ncol = (input_layer_size + 1))
Theta2 = matrix(theta, nrow = num_labels, ncol = (hidden_layer_size + 1))

## ----------------

p1 = predict()
print(paste('accuracy ', mean((p1 == y)) * 100))

