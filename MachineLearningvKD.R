
#installing packages and stuff 
install.packages("xgboost")
library(xgboost)
library(readr)
library(stringr)
install.packages("caret")
library(caret)
library(car)
install.packages('Matrix')
library('Matrix')
library(caret)
allstate_full = read_csv("C:/Users/Kimbe/Documents/OneDrive/Documents/MSA/Machine Learning/Project/allstate_train.csv")
str(allstate_full)

#removing cat116 - because it has 300+ levels and doesn't improve MAE 

allstate_full = allstate_full[ ,c(1:116,118:132)]
allstate_full

#data review 

hist(allstate_full$loss)
#based on above histogram, need to take the log transform 

allstate_full$loss2 = log(allstate_full$loss)
hist(allstate_full$loss2)

#######################################################################################################

#split data set into training and validation into 70/30

splitallstate <- sample(c(T,F), size=nrow(allstate_full), prob=c(0.7,0.3), replace = TRUE)
allstate_train <- allstate_full[splitallstate==T,]
allstate_valid <- allstate_full[splitallstate==F,]

#alternative way to split data set into training and validation into 70/30
allstate_trainindex <- createDataPartition(allstate_full$id, p = .7, list = FALSE, times = 1)
allstate_train = allstate_full[allstate_trainindex, ]
allstate_valid = allstate_full[-allstate_trainindex, ]

#creating sparse models 

sparse_allstatetrain = sparse.model.matrix(loss2 ~ . -id -loss -loss2, data=allstate_train)
sparse_allstatevalid = sparse.model.matrix(loss2 ~ . -id -loss -loss2, data=allstate_valid)
#train_label = as.numeric(levels(train$digit))[train$digit]


#running the xgboost on the sparse models
#tune and run the model
xgb <- xgboost(data = sparse_allstatetrain,
                    label = allstate_train$loss2,
                    eta = 0.02,
                    max_depth = 2,
                    gamma = 6,
                    nround=10000,
                    subsample = 0.8,
                    colsample_bytree = 0.8,
                    objective = "reg:linear",
                    nthread = 3,
                    eval_metric = 'mae',
                    verbose = 1)
ptrain = exp(predict(xgb, sparse_allstatetrain))
pvalid = exp(predict(xgb, sparse_allstatevalid))

#calculating MAE (hopefully)
MAEtrain = sum(abs(allstate_train$loss - ptrain))/length(ptrain)
MAEvalid = sum(abs(allstate_valid$loss - pvalid))/length(pvalid)

#######################################################################################################

#Creating Additional XGBoost Models with smaller number of variables 

#keeping the following - based on initial decision tree - 20 layers: 

allstate_full1 = allstate_full[ ,c("id",
                                  "cat100",
                                  "cat103",
                                  "cat111",
                                  "cat112",
                                  "cat53",
                                  "cat57", 
                                  "cat50", 
                                  "cat80",
                                  "cat81",
                                  "cat87", 
                                  "cont2", 
                                  "cont6", 
                                  "cont7", 
                                  "loss"
                                  )]

allstate_full1

#data review and log transform of target variable  

hist(allstate_full1$loss)
allstate_full1$loss2 = log(allstate_full1$loss)
hist(allstate_full1$loss2)

#split data set into training and validation into 70/30

splitallstate <- sample(c(T,F), size=nrow(allstate_full1), prob=c(0.7,0.3), replace = TRUE)
allstate_train1 <- allstate_full1[splitallstate==T,]
allstate_valid1 <- allstate_full1[splitallstate==F,]

#creating sparse models 

sparse_allstatetrain1 = sparse.model.matrix(loss2 ~ . -id -loss -loss2, data=allstate_train1)
sparse_allstatevalid1 = sparse.model.matrix(loss2 ~ . -id -loss -loss2, data=allstate_valid1)

# tune and run the model
xgb <- xgboost(data = sparse_allstatetrain1,
               label = allstate_train1$loss2,
               eta = 0.05,
               max_depth = 5,
               gamma = 10,
               nround=200,
               subsample = 0.6,
               colsample_bytree = 0.6,
               objective = "reg:linear",
               nthread = 3,
               eval_metric = 'mae',
               verbose = 1)

ptrain1 = exp(predict(xgb, sparse_allstatetrain1))
pvalid1 = exp(predict(xgb, sparse_allstatevalid1))

#calculating MAE (hopefully)
MAEtrain1 = sum(abs(allstate_train1$loss - ptrain1))/length(ptrain1)
MAEvalid1 = sum(abs(allstate_valid1$loss - pvalid1))/length(pvalid1)

#######################################################################################################

#####Subsetting based on the variables from Random Forest (Alec)##############################

allstate_full2 = allstate_full[ ,c("id",
                                   "cat80",
                                   "cont2",
                                   "cat79",
                                   "cat81",
                                   "cat100",
                                   "cat101", 
                                   "cont7", 
                                   "cont12",
                                   "cat111",
                                   "cat12", 
                                   "cat1", 
                                   "cat114", 
                                   "cat103",
                                   "cat53", 
                                   "cat87", 
                                   "cat94", 
                                   "cont14", 
                                   "cat4", 
                                   "cat38", 
                                   "cat57", 
                                   "cat108", 
                                   "cont11", 
                                   "cat72", 
                                   "cat2", 
                                   "cat5", 
                                   "loss"
)]

allstate_full2

#data review and log transform of target variable  

hist(allstate_full2$loss)
allstate_full2$loss2 = log(allstate_full2$loss)
hist(allstate_full2$loss2)

#split data set into training and validation into 70/30

splitallstate <- sample(c(T,F), size=nrow(allstate_full2), prob=c(0.7,0.3), replace = TRUE)
allstate_train2 <- allstate_full2[splitallstate==T,]
allstate_valid2 <- allstate_full2[splitallstate==F,]

#creating sparse models 

sparse_allstatetrain2 = sparse.model.matrix(loss2 ~ . -id -loss -loss2, data=allstate_train2)
sparse_allstatevalid2 = sparse.model.matrix(loss2 ~ . -id -loss -loss2, data=allstate_valid2)

# tune and run the model


xgb <- xgboost(data = sparse_allstatetrain2,
               label = allstate_train2$loss2,
               eta = 0.02,
               max_depth = 3,
               gamma = 6,
               nround=10000,
               subsample = 0.8,
               colsample_bytree = 0.8,
               objective = "reg:linear",
               nthread = 3,
               eval_metric = 'mae',
               verbose = 1)

ptrain2 = exp(predict(xgb, sparse_allstatetrain2))
pvalid2 = exp(predict(xgb, sparse_allstatevalid2))

#calculating MAE (hopefully)
MAEtrain2 = sum(abs(allstate_train2$loss - ptrain2))/length(ptrain2)
MAEvalid2 = sum(abs(allstate_valid2$loss - pvalid2))/length(pvalid2)

#review of the correlation matrix to identify any issues 
correlation = cor(as.matrix(sparse_allstatetrain2), use = "everything")

