# Digit recognition using glm

## set wd
setwd("/Users/snowleopard/Documents/workspace/kaggle/digit-recognition/R/glm")

## clear workspace
closeAllConnections();
rm(list=ls())

## vars
numTree = 50
numRowsForModel = 22000

## load data
train <- read.csv('../../data/train.csv')
smallTrain = train[sample(1:nrow(train), size = numRowsForModel),]
rm(train)
labels = as.factor(smallTrain[[1]])
smallTrain = smallTrain[,-1]

inMyTrain = sample(c(TRUE, FALSE), size = numRowsForModel, replace = TRUE)
myTrain = smallTrain[inMyTrain,]
myTest = smallTrain[!inMyTrain,]
labelsMyTrain = labels[inMyTrain]
labelsMyTest = labels[!inMyTrain]

## train model
library(randomForest)
set.seed(0)
rf <- randomForest(myTrain, labelsMyTrain, ntree = numTree, xtest = myTest, proximity = TRUE)
predictions <- levels(labels)[rf$test$predicted]
predictionIsCorrect = labelsMyTest == predictions
cat(sprintf("Proportion correct in my test set: %f\n", mean(predictionIsCorrect)))
