#---------------------------------Drafts---------------------------------------------------------

# KNN : Improve Model Performance : Testing alternative values of k
i = 1
k_optimum = 1
for (i in 1:200){
  wines_knn_model <-  knn(train=train_arg, test=test_arg, 
                          cl=train_labels, k = i)
  k_optimum[i] <- sum(test_labels == wines_knn_model)/NROW(test_labels)*100
  k = i  
  cat(k,'=',k_optimum[i],'\n')
}
plot(k_optimum, type="l", main="Accuracy with different k-values",
     col="dodgerblue3", xlab = "K-Value", ylab = "% of Accuracy", lwd=2)
legend("bottomright", legend = c("kNN Accuracy"), col= c("dodgerblue3"), lwd=2)

max(k_optimum)
(k_max <- which(k_optimum == max(k_optimum)))

knn_pred_optimum <- knn(train = train_arg, test = test_arg,
                        cl = train_labels, k=k_max, prob = T)

(knn_confmat_optimum <- confusionMatrix(knn_pred_optimum, reference = test_labels))

knn_prob_o <- attr(knn_pred_optimum, "prob")


# Improve Model Performance : Testing alternative values of Trials for Decision Tree
i = 1
t_optimum = 1
for (i in 1:30){
  wines_tree_model <-  C5.0(train[-12], train$label, 
                            trials = i, 
                            weights = NULL, costs = NULL, rules = rules)
  tree_pred <- predict(wines_tree_model, test, type = "class")
  
  t_optimum[i] <- sum(test_labels == tree_pred)/NROW(test_labels)*100
  t = i  
  cat(t,'=',t_optimum[i],'\n')
}
plot(t_optimum, type="l", col="dodgerblue3", xlab = "Trials-Value", ylab = "% of Accuracy", lwd=2)

max(t_optimum)
(t_max <- which(t_optimum == max(t_optimum)))

tree_optimum <- C5.0(train[-12], train$label, trials = t_max, rules = rules)
tree_pred_optimum <- predict(tree_optimum, test, type = "class")
confusionMatrix(tree_pred_optimum, reference = test$label)

tree_prob_optimum <- predict(tree_optimum, test, type = "prob")[,2]