# Digit recognition using glm

## set wd
setwd("/Users/snowleopard/Documents/workspace/kaggle/digit-recognition/R/glm")

## clear workspace
closeAllConnections();
rm(list=ls())

## vars
numTrees = 25

## load data
print('Loading data...')
data <- read.csv('../../data/train.csv')
trainSample = sample(c(TRUE,FALSE), size=(nrow(data)*0.7), replace=T)
train = data[trainSample,]
trainTest = data[!trainSample,]

# labels = as.factor(data[,1])
trainLabel = as.factor(train[,1])
trainTestLabel = as.factor(trainTest[,1])
train = train[,-1]
trainTest = trainTest[,-1]

test <- read.csv('../../data/test.csv')
# rows <- sample(1:nrow(data), numRowsForModel)

# train = data[,-1]

## run randomforest
print('Running RandomForest...')
library(randomForest)
rf <- randomForest(train, trainLabel, ntree=numTrees, xtest=trainTest, ytest = trainTestLabel, keep.forest = T)
predictions <- levels(trainLabel)[rf$test$predicted]
predictionIsCorrect = trainTestLabel == predictions
print(paste('proportion correc', mean(predictionIsCorrect)))
# head(predictions)

predicted <- predict(rf, newdata = test, OOB=T, type="response")
predictions <- data.frame(ImageId=1:nrow(test),Label=levels(trainLabel)[predicted])
write.csv(predictions, 'rf_benchmark.csv', row.names=FALSE, quote=FALSE)
