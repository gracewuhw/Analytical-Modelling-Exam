library("ggplot2")
library("caret") # split the data
library("glmnet") # lasso
library("pROC") # draw ROC
library("rpart") # decision tree
library("rpart.plot") # draw decision tree
library("party") # conditional inference tree
library("purrr") # map_chr
library("randomForest")
library("e1071") # SVM

# import the data
setwd("D://ESCP Europe//Analytical Modelling//EXAM")
creditdata <- read.csv("Exam2019- Dataset.csv", header = TRUE)
head(creditdata)
# check the data type
str(creditdata)
# check if there are any missing values
sapply(creditdata, function(x) sum(is.na(x)))
# format data
creditdata$Default <- as.factor(creditdata$Default)
creditdata$cards <- as.factor(creditdata$cards)
str(creditdata)

# explore continous data
attach(creditdata) 
summary(duration)
brksCredit <- seq(0, 80, 10) 
hist(duration, breaks=brksCredit, xlab = "Credit Month", ylab = "Frequency", main = " ", cex=0.4)
boxplot(duration, bty="n",xlab = "Credit Month", cex=0.4)

summary(amount)
hist(amount, xlab = "Credit Amount", ylab = "Frequency", main = " ", cex=0.4) # produces nice looking histogram
boxplot(amount, bty="n",xlab = "Credit Amount", cex=0.4)

summary(age)
hist(age, breaks=brksCredit, xlab = "Age", ylab = "Frequency", main = " ", cex=0.4) # produces nice looking histogram
boxplot(age, bty="n",xlab = "Age", cex=0.4)

ggplot(creditdata, aes(factor(installment)))+ geom_bar(aes(fill = Default), position = "dodge") + xlab("Installment Rates")
qplot(Default,amount, data = creditdata, geom = "boxplot")
qplot(Default,age, data = creditdata, geom = "boxplot")

ggplot(creditdata, aes(duration, fill=Default))+ geom_density(alpha=.5) 
qplot(Default,residence, data = creditdata, geom = "boxplot")

# explore categorical data
ggplot(creditdata, aes(checkingstatus1, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(history, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(purpose, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(savings, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(others, ..count..)) + geom_bar(aes(fill = Default), position = "dodge")  
ggplot(creditdata, aes(status, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(property, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(otherplans, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(housing, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(job, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(foreign, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(employ, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 
ggplot(creditdata, aes(tele, ..count..)) + geom_bar(aes(fill = Default), position = "dodge") 

# split the data
set.seed(12420424)
in.train <- createDataPartition(creditdata$Default, p=0.8, list=FALSE)
german_credit.train <- creditdata[in.train,]
german_credit.test <- creditdata[-in.train,]

# variable selection
# Stepwise variable selection using AIC
credit.glm0 <- glm(Default ~ ., family = binomial, german_credit.train)
credit.glm.step <- step(credit.glm0, direction = "backward")
summary(credit.glm.step)

# Stepwise variable selection using BIC
credit.glm.step.bic <- step(credit.glm0, k = log(nrow(german_credit.train)))
summary(credit.glm.step.bic)

# Lasso variable selection
# To get variable selection using LASSO, we first create matrix of the dataset.
factor_var <- c(1,2,4,5,7,8,10,11,13,15,16,17,18,20,21)
num_var <- c(3,6,9,12,14,19)
train2 <- german_credit.train
train2[num_var] <- scale(train2[num_var])
train2[factor_var] <- sapply(train2[factor_var] , as.numeric)
X.train <- as.matrix(train2[,2:21])
Y.train <- as.matrix(train2[,1])

# We fit the LASSO model to our data. From the plot below, we see that as the value of lambda keeps on increasing, the coefficients for the variables tend to 0.
lasso.fit<- glmnet(x=X.train, y=Y.train, family = "binomial", alpha = 1)
plot(lasso.fit, xvar = "lambda", label=TRUE)

# Using cross validation to find perfect lambda value
cv.lasso<- cv.glmnet(x=X.train, y=Y.train, family = "binomial", alpha = 1, nfolds = 10)
plot(cv.lasso)
cv.lasso$lambda.1se
coef(lasso.fit, s=cv.lasso$lambda.1se)

# Model 1 - Logistic regression
# create the model
credit.glm.final <- glm(Default ~ checkingstatus1 + duration + history + amount + savings + otherplans, family = binomial, german_credit.train)
summary(credit.glm.final)
# predict with test dataset
lr.pred <- predict(credit.glm.final, german_credit.test, type = "response")
lr.pred1 <- lr.pred > 0.5
lr.pred1 <- as.numeric(lr.pred1)
# draw confusion matrix
lr.pref <- table(german_credit.test$Default, lr.pred1, dnn = c("Actual", "Predicted")) 
lr.pref
# draw ROC curve
lrdata <- data.frame(prob=lr.pred, obs=german_credit.test$Default)
roc(lrdata$obs, lrdata$prob, plot=TRUE, print.thres=TRUE, print.auc=TRUE)

# Model 2 - Decision Tree
# create the model
set.seed(1234)
dtree <- rpart(Default ~ ., data = german_credit.train, method = "class", parms = list(split = "information"))
dtree$cptable
plotcp(dtree)
dtree.pruned <- prune(dtree, cp=.025)
prp(dtree.pruned, type = 2, extra = 104, fallen.leaves = TRUE, main="Decision Tree") 
# prediction
dtree.pred <- predict(dtree.pruned, german_credit.test, type="class")
# confusion matrix
dtree.pref <- table(german_credit.test$Default, dtree.pred, dnn=c("Actual", "Predicted"))
dtree.pref
# ROC 
dtree.pred <- predict(dtree.pruned, german_credit.test, type="prob")
dtdata <- data.frame(prob=dtree.pred[,2], obs=german_credit.test$Default)
roc(dtdata$obs, dtdata$prob, plot=TRUE, print.thres=TRUE, print.auc=TRUE)

# Model 3 - Conditional Inference tree
# create the model
fit.ctree <- ctree(Default~., data=german_credit.train)
plot(fit.ctree, main="Conditional Inference Tree")
# prediction
ctree.pred <- predict(fit.ctree, german_credit.test, type="response")
# comfusion matrix
ctree.table <- table(german_credit.test$Default, ctree.pred,dnn=c("Actual", "Predicted"))
ctree.table
# ROC
ctree.pred <- predict(fit.ctree, german_credit.test, type="prob")
ctreeprob <- as.data.frame(map_chr(ctree.pred,2))
ctreedata <- data.frame(prob=ctreeprob[,1], obs=german_credit.test$Default)
ctreedata$prob <- as.numeric(ctreedata$prob)
roc(ctreedata$obs, ctreedata$prob, plot=TRUE, print.thres=TRUE, print.auc=TRUE)

# Model 4 - Random Forest
# create the model
set.seed(1234)
fit.forest <- randomForest(Default~., data=german_credit.train,importance=TRUE)
fit.forest 
importance(fit.forest, type=2)
# perdiction
forest.pred <- predict(fit.forest, german_credit.test, type = "response")
# confusion matrix
forest.pref <- table(german_credit.test$Default, forest.pred, dnn=c("Actual", "Predicted")) 
forest.pref
# ROC
forest.pred <- predict(fit.forest, german_credit.test, type = "prob")
rfdata <- data.frame(prob=forest.pred[,2], obs=german_credit.test$Default)
roc(rfdata$obs, rfdata$prob, plot=TRUE, print.thres=TRUE, print.auc=TRUE)

# Model 5 - SVM
# create the model
set.seed(1234)
fit.svm <- svm(Default~., data=german_credit.train, probability=TRUE)
fit.svm
# perdiction
svm.pred <- predict(fit.svm, na.omit(german_credit.test), probability=TRUE)
# confusion matrix
svm.pref <- table(na.omit(german_credit.test)$Default, svm.pred, dnn=c("Actual", "Predicted"))
svm.pref
# ROC
svm.prob <- attr( svm.pred, "probabilities")
svmdata <- data.frame(prob=svm.prob[,2], obs=german_credit.test$Default)
roc(svmdata$obs, svmdata$prob, plot=TRUE, print.thres=TRUE, print.auc=TRUE)

# Model 6 - SVM with BFB
# create the model
set.seed(1234)
tuned <- tune.svm(Default~., data=german_credit.train,gamma=10^(-6:1),cost=10^(-10:10))
tuned
fit.svmbfb <- svm(Default~., data=german_credit.train, gamma=.01, cost=10, probability = TRUE)
# prediction
svmbfb.pred <- predict(fit.svmbfb, na.omit(german_credit.test), probability = TRUE)
# confusion matrix
svmbfb.pref <- table(na.omit(german_credit.test)$Default, svmbfb.pred, dnn=c("Actual", "Predicted")) 
svmbfb.pref
# ROC
svmbfb.pred <- attr( svmbfb.pred, "probabilities")
svmbfbdata <- data.frame(prob=svmbfb.pred[,2], obs=german_credit.test$Default)
roc(svmbfbdata$obs, svmbfbdata$prob, plot=TRUE, print.thres=TRUE, print.auc=TRUE)

# Compare the models
# define the function to show performance
performance <- function(table, n=2){
 if(!all(dim(table) == c(2,2))) 
 	stop("Must be a 2 x 2 table")
 tn = table[1,1]
 fp = table[1,2]
 fn = table[2,1]
 tp = table[2,2]
 sensitivity = tp/(tp+fn)
 specificity = tn/(tn+fp)
 ppp = tp/(tp+fp)
 npp = tn/(tn+fn)
 hitrate = (tp+tn)/(tp+tn+fp+fn)
 f1 = 2*(ppp*sensitivity)/(ppp+sensitivity)
 result <- paste("Sensitivity/Recall = ", round(sensitivity, n) ,
 	"\nSpecificity = ", round(specificity, n),
 	"\nPrecision = ", round(ppp, n),
 	"\nNegative Predictive Value = ", round(npp, n),
 	"\nF1-score = ", round(f1, n),
 	"\nAccuracy = ", round(hitrate, n), "\n", sep="")
 cat(result)
} 
# print the result
performance(svmbfb.pref) 
performance(svm.pref) 
performance(forest.pref)
performance(ctree.table)
performance(dtree.pref)
performance(lr.pref)