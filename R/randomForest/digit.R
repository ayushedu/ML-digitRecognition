# Digit recognition using glm

## set wd
setwd("/Users/snowleopard/Documents/workspace/kaggle/digit-recognition/R/glm")

## clear workspace
closeAllConnections();
rm(list=ls())

## vars
numTrees = 25
numRowsForModel = 10000

## load data
print('Loading data...')
data <- read.csv('../../data/train.csv')
test <- read.csv('../../data/test.csv')
#rows <- sample(1:nrow(data), numRowsForModel)
labels = as.factor(data[,1])
train = data[,-1]

## run randomforest
print('Running RandomForest...')
library(randomForest)
rf <- randomForest(train, labels, ntree=numTrees, xtest=test)
predictions <- data.frame(Label=levels(labels)[rf$test$predicted])
head(predictions)

write.csv(predictions, 'rf_benchmark.csv', row.names=FALSE)
