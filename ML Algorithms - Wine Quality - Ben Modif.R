################################# MACHINE LEARNING PROJECT ##########################################
rm(list = ls())

# Load packages
if(FALSE){
  install.packages("devtools")
  install.packages("rpart")
  install.packages("rpart.plot")
  install.packages("neuralnet") 
  install.packages("e1071")
  install.packages("kernlab")
  install.packages("tree")
  install.packages("gmodels")
  install.packages("ggcorrplot")
  install.packages("randomForest")
  install.packages("caret")
  install.packages("C50")
  install.packages("rattle")
  install.packages("ROCR")  
  install.packages("pROC")
}#end of if statement

library(ROCR)
library(pROC)
library(ggplot2)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(e1071)
library(kernlab)
library(reshape2)
library(tree)
library(gmodels)
library(ggcorrplot)
library(randomForest)
library(C50)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#-----------------------------------------------------------------------------------------------------
#### Load database of red wines and white wines ####

# Load database
white <- read.csv(paste(getwd(),"/winequality-white.csv", sep =""), sep=";")#remove at the end
#red <- read.csv(paste(getwd(),"/winequality-red.csv", sep =""), sep=";")#remove at the end

wines <- white

#-----------------------------------------------------------------------------------------------------
#### Data Exploration ####

# Structure of data
str(wines)
head(wines)

# Histogram of each features
par(mfrow=c(2,3))
for(i in 1:6) {
  hist(wines[,i], main=names(wines)[i])
}
for(i in 7:12) {
  hist(wines[,i], main=names(wines)[i])
}


#-----------------------------------------------------------------------------------------------------
#### Correlation Matrix ####

# Check correlation for multicollinearity 
corrmat <- as.matrix(cor(wines[,1:12]))
ggcorrplot(corrmat)


#-----------------------------------------------------------------------------------------------------
#### Convert Quality into Factors : Bad / Good ####

# Convert wines quality into factors : Bad / Good
par(mfrow=c(1,1))
hist(wines$quality, col="dodgerblue3", main="Distribution of Wine Quality")
table(wines$quality)

wines_f <- wines
for (i in 1:nrow(wines_f)) {
  if (wines_f$quality[i] >= 6)
    wines_f$label[i] <- 1
  else if (wines_f$quality[i] < 6)
    wines_f$label[i] <- 0
}
wines_f$label <- factor(wines_f$label, levels = c(1,0), labels = c("Good","Bad"))
table(wines_f$label)

# Removing the quality variable
wines_f$quality <- NULL

#-----------------------------------------------------------------------------------------------------
#### Normalization of all variables ####

# Function to normalize data
normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}
wines_n <- as.data.frame(lapply(wines_f[,1:11], normalize))
wines_n$label <- wines_f$label

#-----------------------------------------------------------------------------------------------------
#### Create Training and Testing Datasets ####

# Separation: 80% Training Set & 20% Testing Set
pct_training <- 0.80

(n_train <- floor(pct_training*nrow(wines_n)))
(n_test <- nrow(wines_n) - n_train)

train <- wines_n[1:n_train, ]
test <- wines_n[(n_train+1):(nrow(wines_n)), ]

train_labels <- train[,12]
test_labels <- test[,12]

#-----------------------------------------------------------------------------------------------------
#### Machine Learning Algorithm : kNN - Classification ####

# Double check the structure
str(wines_n)
round(prop.table(table(wines_n$label)) * 100, digits = 1)
sum(is.na(wines_n))

# Remove Quality Labels
train_arg <- train[,1:11]
test_arg <- test[,1:11]

# kNN Model
n <- 200
wines_knn_pred <- list()
knn_confmatrix <- list()
for(i in 1:n){
  print(i)
  wines_knn_pred[[i]] <- knn(train = train_arg, test = test_arg, cl = train_labels, k=i, prob = T)
  knn_confmatrix[[i]] <- confusionMatrix(wines_knn_pred[[i]], reference = test_labels)
}

# Create Matrix of results for all models
knn_FNFP <- matrix(NA, nrow = n, ncol = 3)
colnames(knn_FNFP) <- c("Accuracy","FP (Type I error)","FN (Type II error)")
for(i in 1:n){
  knn_FNFP[i,1] <- round(knn_confmatrix[[i]]$overall[1],3)
}
for(i in 1:n){
  knn_FNFP[i,2] <- knn_confmatrix[[i]]$table[1,2]
}
for(i in 1:n){
  knn_FNFP[i,3] <- knn_confmatrix[[i]]$table[2,1]
}
knn_FNFP

# Compute most accurate / Min FP / Min FN
(knn_max_acc <- min(which(knn_FNFP == max(knn_FNFP[,1]))))
(knn_min_FP <- min(knn_FNFP[,2]))
(knn_min_FP <- which(knn_FNFP==knn_min_FP, arr.ind=TRUE)[1])
(knn_min_FN <- min(knn_FNFP[,3]))
(knn_min_FN <- which(knn_FNFP==knn_min_FN, arr.ind=TRUE)[1])

# ROC Curve for the three above models
knn_max_acc_prob <- attr(wines_knn_pred[[knn_max_acc]], "prob")
knn_FP_prob <- attr(wines_knn_pred[[knn_min_FP]], "prob")
knn_FN_prob <- attr(wines_knn_pred[[knn_min_FN]], "prob")

# ROC Curve (Receiver Operating Characteristic) & AUC
par(pty="s")
roc(response = test_labels, predictor =  knn_max_acc_prob, main="kNN with variable k values",
    plot=T,legacy.axes=T,percent=T,print.auc=T,
    xlab="% False Positive", ylab="% True Positive", col="dodgerblue3",lwd = 2)
roc(response = test_labels, predictor = knn_FP_prob, plot = T, percent = T, print.auc = T,
    col="#4daf4a", lwd=2, add=T, print.auc.y=40)
roc(response = test_labels, predictor = knn_FN_prob , plot = T, percent = T, print.auc = T,
    col="orange", lwd=2, add=T, print.auc.y=30)
legend("bottomright", legend = c("Max Accuracy", "Min False Positive", "Min False Negative"), 
       col= c("dodgerblue3","#4daf4a", "orange"), lwd=2)

# Improve Model Performance : z-score standardization
wines_z <- as.data.frame(scale(wines[-12]))

train_z <- wines_z[1:n_train, ]
test_z <- wines_z[(n_train+1):(nrow(wines_z)), ]

wines_knn_pred_z <- knn(train = train_z, test = test_z,
                        cl = train_labels, k = 63) #sqrt(number of rows)

(knn_confmat_z <- confusionMatrix(wines_knn_pred_z, reference = test_labels))
(knn_acc_z <- knn_confmat_z$overall[1])


#-----------------------------------------------------------------------------------------------------
#### Machine Learning Algorithm : Classification with Decision Trees ####

# Define if Classification Tree or Rule-based Tree
rules <- FALSE

# C5.0 Algorithm : classification Tree or Rule-based Tree
c <- 100
tree <- list()
tree_pred <- list()
tree_confmatrix <- list()
for(i in 1:c){
  print(i)
  tree[[i]] <- C5.0(train[-12], train$label, trials = i, weights = NULL, costs = NULL, rules = rules)
  tree_pred[[i]] <- predict(tree[[i]], test, type = "class")
  tree_confmatrix[[i]] <- confusionMatrix(tree_pred[[i]], reference = test_labels)
}
tree_confmatrix[[1]]
tree_confmatrix[[10]]

# Create Matrix of results for all models
tree_FNFP <- matrix(NA, nrow = c, ncol = 3)
colnames(tree_FNFP) <- c("Accuracy","FP (Type I error)","FN (Type II error)")
for(i in 1:c){
  tree_FNFP[i,1] <- round(tree_confmatrix[[i]]$overall[1],3)
}
for(i in 1:c){
  tree_FNFP[i,2] <- tree_confmatrix[[i]]$table[1,2]
}
for(i in 1:c){
  tree_FNFP[i,3] <- tree_confmatrix[[i]]$table[2,1]
}
tree_FNFP

# Compute most accurate / Min FP / Min FN
(tree_max_acc <- min(which(tree_FNFP == max(tree_FNFP[,1]))))
(tree_min_FP <- min(tree_FNFP[,2]))
(tree_min_FP <- which(tree_FNFP==tree_min_FP, arr.ind=TRUE)[1])
(tree_min_FN <- min(tree_FNFP[,3]))
(tree_min_FN <- which(tree_FNFP==tree_min_FN, arr.ind=TRUE)[1])

# Compute predictions for most accurate / Min FP / Min FN models with Probabilities Output
tree_max_acc <- predict(tree[[tree_max_acc]], test, type = "prob")
tree_max_acc <- tree_max_acc[,2]
tree_min_FP <- predict(tree[[tree_min_FP]], test, type = "prob")
tree_min_FP <- tree_min_FP[,2]
tree_min_FN <- predict(tree[[tree_min_FN]], test, type = "prob")
tree_min_FN <- tree_min_FN[,2]

# ROC Curve (Receiver Operating Characteristic) & AUC
par(pty="s")
roc(response = test_labels, predictor = tree_max_acc , main=" Decision Trees with  variable Trials",
    plot=T,legacy.axes=T,percent=T,print.auc=T,
    xlab="% False Positive", ylab="% True Positive", col="dodgerblue3",lwd = 2)
roc(response = test_labels, predictor = tree_min_FP, plot = T, percent = T, print.auc = T,
    col="#4daf4a", lwd=2, add=T, print.auc.y=40)
roc(response = test_labels, predictor = tree_min_FN , plot = T, percent = T, print.auc = T,
    col="orange", lwd=2, add=T, print.auc.y=30)
legend("bottomright", legend = c("Max Accuracy", "Min False Positive", "Min False Negative"), 
       col= c("dodgerblue3","#4daf4a", "orange"), lwd=2)


#-----------------------------------------------------------------------------------------------------
#### Machine Learning Algorithm : Decision Trees : Recursive Partitioning ####

# Recursive Partitioning Algo
m <- 10
cp <- 0.01
model_rpart <- list()
rpart_pred <- list()
rpart_confmatrix <- list()
rpart_acc <- 1

for(i in 1:m){
  print(i)
  model_rpart[[i]] <- rpart(label ~ ., data = train, method = "class",
                            control = rpart.control(minsplit = i, cp = cp)) 
  rpart_pred[[i]] <- predict(model_rpart[[i]], test, type = "class")
  rpart_confmatrix[[i]] <- confusionMatrix(rpart_pred[[i]], reference = test_labels)
  rpart_acc[i] <- sum(test_labels == rpart_pred[[i]])/NROW(test_labels)*100
  rp = i  
  cat(rp,'=',rpart_acc[i],'\n')
}

# Recursive Partitioning Plot
rpart.plot(model_rpart[[10]], digits = 3)
rpart.plot(model_rpart[[10]], digits = 3, fallen.leaves = TRUE,
           type = 3, extra = 106)
fancyRpartPlot(model_rpart[[10]], caption = NULL)

# Create Matrix of results for all models
rpart_FNFP <- matrix(NA, nrow = m, ncol = 3)
colnames(tree_FNFP) <- c("Accuracy","FP (Type I error)","FN (Type II error)")
for(i in 1:m){
  rpart_FNFP[i,1] <- round(rpart_confmatrix[[i]]$overall[1],3)
}
for(i in 1:m){
  rpart_FNFP[i,2] <- rpart_confmatrix[[i]]$table[1,2]
}
for(i in 1:m){
  rpart_FNFP[i,3] <- rpart_confmatrix[[i]]$table[2,1]
}
rpart_FNFP


#-----------------------------------------------------------------------------------------------------
#### Machine Learning Algorithm : Random Forest ####

# RandomForest Model
r <- 10
rf <- list()
rf_pred <- list()
rf_prob <- list()
rf_confmatrix <- list()
for(i in 1:r){
  print(i)
  rf[[i]] <- randomForest(label ~ ., data = train, mtry=r)
  rf_pred[[i]] <- predict(rf[[i]], test, type = "class")
  rf_prob[[i]] <- predict(rf[[i]], test, type = "prob")
  rf_confmatrix[[i]] <- confusionMatrix(rf_pred[[i]], reference = test_labels)
}

# Create Matrix of results for all models
rf_FNFP <- matrix(NA, nrow = r, ncol = 3)
colnames(rf_FNFP) <- c("Accuracy","FP (Type I error)","FN (Type II error)")
for(i in 1:r){
  rf_FNFP[i,1] <- round(rf_confmatrix[[i]]$overall[1],3)
}
for(i in 1:r){
  rf_FNFP[i,2] <- rf_confmatrix[[i]]$table[1,2]
}
for(i in 1:r){
  rf_FNFP[i,3] <- rf_confmatrix[[i]]$table[2,1]
}
rf_FNFP

# Compute most accurate / Min FP / Min FN
(rf_max_acc <- min(which(rf_FNFP == max(rf_FNFP[,1]))))
(rf_min_FP <- min(rf_FNFP[,2]))
(rf_min_FP <- min(which(rf_FNFP==rf_min_FP, arr.ind=TRUE)[1]))
(rf_min_FN <- min(rf_FNFP[,3]))
(rf_min_FN <- min(which(rf_FNFP==rf_min_FN, arr.ind=TRUE)[1]))

# Compute predictions for most accurate / Min FP / Min FN models with Probabilities Output
rf_max_acc <- predict(rf[[rf_max_acc]], test, type = "prob")
rf_max_acc <- rf_max_acc[,2]
rf_min_FP <- predict(rf[[rf_min_FP]], test, type = "prob")
rf_min_FP <- rf_min_FP[,2]
rf_min_FN <- predict(rf[[rf_min_FN]], test, type = "prob")
rf_min_FN <- rf_min_FN[,2]

# ROC Curve (Receiver Operating Characteristic) & AUC
par(pty="s")
roc(response = test_labels, predictor = rf_max_acc , main=" RandomForest with variable Mtry",
    plot=T,legacy.axes=T,percent=T,print.auc=T,
    xlab="% False Positive", ylab="% True Positive", col="dodgerblue3",lwd = 2)
roc(response = test_labels, predictor = rf_min_FP, plot = T, percent = T, print.auc = T,
    col="#4daf4a", lwd=2, add=T, print.auc.y=40)
roc(response = test_labels, predictor = rf_min_FN , plot = T, percent = T, print.auc = T,
    col="orange", lwd=2, add=T, print.auc.y=30)
legend("bottomright", legend = c("Max Accuracy", "Min False Positive", "Min False Negative"), 
       col= c("dodgerblue3","#4daf4a", "orange"), lwd=2)


#-----------------------------------------------------------------------------------------------------
#### Machine Learning Algorithm : ANN MLP Model ####

# ANN MLP Model
model.mlp = list()
a = 1;
for(i in 1:1){
  for(j in 0:0){
    print(paste("i = ", i, "j = ", j))
    if(j == 0){model.mlp[[a]] = neuralnet(label ~.,data = wines_n, hidden = i, linear.output = FALSE)}#end of if statement
    else{model.mlp = neuralnet(label ~.,data = wines_n, hidden = c(i,j), linear.output = FALSE)}#end of else statement
    a = a+1;#increment a
  }#end of second for loop
}#end of first for loop

#plot ANN
par(pty="s")
plot.nnet(model.mlp[[1]])
plot(model.mlp[[1]])

# Table predictions & accuracy
model.mlp.results <- predict(model.mlp[[1]], test[1:11])
model.mlp.predicted.label = ifelse(model.mlp.results[,2]>0.5,"Good","Bad")
model.mlp.predicted.label = data.frame(keyName=names(model.mlp.predicted.label), value=model.mlp.predicted.label, row.names=NULL)
confusionMatrix(model.mlp.predicted.label[,2], reference = test$label)

model.mlp_prob <- predict(model.mlp, test, type ="prob")[,2]
# ROC & AUC
roc(response = test_labels, predictor = model.mlp_prob, plot = T, legacy.axes = T, 
    percent = T, xlab="% False Positive", ylab="% True Positive", print.auc=T,
    col="red",lwd = 4)
legend("bottomright", legend = c("ANN MLP"), 
       col= c("red"), lwd=4)


#-----------------------------------------------------------------------------------------------------
#### ROC & AUC Comparison between Models ####

# Plot most accurate of kNN 
roc(response = test_labels,predictor=knn_max_acc_prob, plot = T,legacy.axes = T,percent=T,print.auc=T,
    xlab="% False Positive", ylab="% True Positive", 
    col="dodgerblue3",lwd = 2, print.auc.x=55, print.auc.y=55)

# Plot most accurate of Classification trees
roc(response = test_labels,predictor=tree_max_acc,plot = T,percent = T,print.auc = T, add=T,
    col="#4daf4a", lwd=2, print.auc.x=55, print.auc.y=47)

# Plot most accurate of Random Forest
roc(response = test_labels, predictor=rf_max_acc, plot = T,percent = T, print.auc = T, add=T,
    col="orange", lwd=2, print.auc.x=25, print.auc.y=55)

# Plot most accurate of AAN
roc(response = test_labels, predictor = model.mlp_prob, plot = T, 
    percent = T, print.auc=T, add=T,
    col="red",lwd = 2, print.auc.x=25, print.auc.y=47)

legend("bottomright",
       legend = c("kNN","Classification Tree","Random Forest","ANN MLP"), 
       col= c("dodgerblue3","#4daf4a","orange", "red"), lwd=2)


#-----------------------------------------------------------------------------------------------------
#### Table Model Comparisons Results  ####

results <- matrix(NA, nrow = 4, ncol = 4)
colnames(results) <- c("kNN","Tree","RandomForest","ANN MLP")
rownames(results) <- c("False Positive (Type I error)","False Negative (Type II error)",
                       "True Positive","True Negative")
results

(knn_comp <- knn_confmatrix[[knn_max_acc]])
(tree_comp <- tree_confmatrix[[tree_max_acc]])
(rf_comp <- rf_confmatrix[[rf_max_acc]])

results[1,1] <- knn_comp$table[1,2]
results[2,1] <- knn_comp$table[2,1]
results[3,1] <- knn_comp$table[1,1]
results[4,1] <- knn_comp$table[2,1]
results
results[1,2] <- tree_comp$table[1,2]
results[2,2] <- tree_comp$table[2,1]
results[3,2] <- tree_comp$table[1,1]
results[4,2] <- tree_comp$table[2,1]
results
results[1,3] <- rf_comp$table[1,2]
results[2,3] <- rf_comp$table[2,1]
results[3,3] <- rf_comp$table[1,1]
results[4,3] <- rf_comp$table[2,1]
results
results <- as.data.frame((results))

# Statistics of Models
results_s <- matrix(NA, nrow = 3, ncol = 4)
colnames(results_s) <- c("kNN","Tree","RandomForest","ANN MLP")
rownames(results_s) <- c("Accuracy","Sensitivity","Specificity")

results_s[1,1] <- round(knn_comp$overall[1],3)
results_s[2,1] <- round((knn_comp$table[1,1]/(knn_comp$table[1,1]+knn_comp$table[2,1])),3)
results_s[3,1] <- round((knn_comp$table[2,2]/(knn_comp$table[2,2]+knn_comp$table[1,2])),3)

results_s[1,2] <- round(tree_comp$overall[1],3)
results_s[2,2] <- round((tree_comp$table[1,1]/(tree_comp$table[1,1]+tree_comp$table[2,1])),3)
results_s[3,2] <- round((tree_comp$table[2,2]/(tree_comp$table[2,2]+tree_comp$table[1,2])),3)

results_s[1,3] <- round(rf_comp$overall[1],3)
results_s[2,3] <- round((rf_comp$table[1,1]/(rf_comp$table[1,1]+rf_comp$table[2,1])),3)
results_s[3,3] <- round((rf_comp$table[2,2]/(rf_comp$table[2,2]+rf_comp$table[1,2])),3)
results_s

results_s <- as.data.frame((results_s))

#----------------------------- DRAFT -----------------------------------------------------------

conftree <-confusionMatrix(tree_pred_optimum, reference = test$label)
results[1,2] <- conftree$table[1,2]
results[2,2] <- conftree$table[2,1]
results[3,2] <- conftree$table[1,1]
results[4,2] <- conftree$table[2,1]

confrpart <- confusionMatrix(rpar_pred_o, reference = test$label)
results[1,3] <- confrpart$table[1,2]
results[2,3] <- confrpart$table[2,1]
results[3,3] <- confrpart$table[1,1]
results[4,3] <- confrpart$table[2,1]

confrf <-confusionMatrix(rf_pred, reference = test$label)
results[1,4] <- confrf$table[1,2]
results[2,4] <- confrf$table[2,1]
results[3,4] <- confrf$table[1,1]
results[4,4] <- confrf$table[2,1]

confaan <- confusionMatrix(model.mlp.predicted.label[,2], reference = test$label)
results[1,5] <- confaan$table[1,2]
results[2,5] <- confaan$table[2,1]
results[3,5] <- confaan$table[1,1]
results[4,5] <- confaan$table[2,1]

results <- as.data.frame(results)
