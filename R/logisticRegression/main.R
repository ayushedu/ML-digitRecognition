# Digit recognition using logistic regression

## load files
source('sigmoid.R')
source('lrCostFunction.R')
source('lrGradient.R')
source('oneVsAll.R')
source('predictOneVsAll.R')

## set wd
setwd("C:/Users/avatsyay/Documents/personal/kaggle/digitRecognition/R/logisticRegression")

## load data
print('Reading data...')
#  data = read.csv('../../data/train.csv');
#  y = data[,1]
#  X = as.matrix(data[,2:dim(data)[2]])
#  rm(data);

## initialize vars
lambda = 1
num_labels = 10
#initial_theta = stdnormal_rnd(dim(x)[2], num_labels);

all_theta <- oneVsAll(X,y,num_labels, lambda)
pred <- predictOneVsAll(all_theta, X)
print(paste('accuracy ', mean((pred == y)) * 100))

