# Random forest algo for digit recognition

## set dir
setwd('/Users/snowleopard/Documents/workspace/kaggle/digit-recognition/R/randomForest')

## clear workspace
closeAllConnections();
rm(list=ls())

## load libraries and code
library(randomForest)

## load data
print('Loading data...')
data = read.csv('../../data/train.csv');
m <- dim(data)[1]
trainData <- data[1:(m*0.1),]
testData <- data[(m*0.9):m,]
rm(data)

data.rf <- randomForest(label ~ ., data=trainData, importance=TRUE,proximity=TRUE)
