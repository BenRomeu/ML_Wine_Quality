################################# MACHINE LEARNING PROJECT ##########################################
#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
#tutorial for inserting text: https://shiny.rstudio.com/tutorial/written-tutorial/lesson2/
################################# MACHINE LEARNING PROJECT ##########################################
rm(list = ls())

# Load packages
if(FALSE){
  install.packages("shinythemes")
  install.packages("DT")
  install.packages("devtools")
  install.packages("RWeka")
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
  install.packages("knitr")
}#end of if statement

library(ggplot2)
library(knitr)
library(ROCR)
library(pROC)
library(ggplot2)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(caret)
library(class)
library(RWeka)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(e1071)
library(kernlab)
library(tree)
library(gmodels)
library(ggcorrplot)
library(randomForest)
library(C50)
library(devtools)
  source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
library(shiny)
library(shinythemes)
library(gridExtra)
library(grid)
  

  #### Load database of red wines and white wines ####
  load(paste(getwd(),"/data_set.RData", sep = ""), envir = parent.frame(), verbose = FALSE)

#-----------------------------------------------------------------------------------------------------
#### Shiny App

# Define UI for application that draws a histogram
ui <- fluidPage(theme = shinytheme("sandstone"),
   
   # Application title
   titlePanel("Machine Learning Algorithms - How To Predict Wine Quality"),
   
   tabsetPanel(id = "tabSelected",
               tabPanel("Model Explanation", uiOutput("Model_Explanation")),
               tabPanel("Demonstration", uiOutput("Demonstration")),
               tabPanel("Data Analysis", uiOutput("Data_Analysis"))
               
   )#end of tabsetPanel
   
)# end of ui fluidpage

# Define server logic required to draw a histogram
server <- function(input, output) {
  
    output$ML_most_accurate = renderPlot({
      #### ROC & AUC Comparison between Models ####
      #make the plot square
      par(pty="s")
      # Plot most accurate of kNN 
      roc(response = test_labels,predictor=knn_max_acc_prob, plot = T,legacy.axes = T,percent=T,print.auc=T,
          xlab="% False Positive", ylab="% True Positive", 
          col="dodgerblue3",lwd = 2, print.auc.x=55, print.auc.y=55)
      
      # Plot most accurate of Classification trees
            #rule == FALSE
            roc(response = test_labels,predictor=tree_max_acc,plot = T,percent = T,print.auc = T, add=T,
                col="#4daf4a", lwd=2, print.auc.x=55, print.auc.y=47)
            #rule == TRUE
            roc(response = test_labels,predictor=treeT_max_acc,plot = T,percent = T,print.auc = T, add=T,
                col="green", lwd=2, print.auc.x=25, print.auc.y=47)
      
      # Plot most accurate of Random Forest
      roc(response = test_labels, predictor=rf_max_acc, plot = T,percent = T, print.auc = T, add=T,
          col="orange", lwd=2, print.auc.x=25, print.auc.y=55)
      
      # Plot most accurate of AAN
      #roc(response = test_labels, predictor = ANN_max_acc_prob, plot = T, 
      #    percent = T, print.auc=T, add=T,
      #    col="red",lwd = 2, print.auc.x=25, print.auc.y=47)
      
      legend("bottomright",
             legend = c(paste("knn (k = ",knn_max_acc_i,")"),"Classification Tree (rule = FALSE)",
                        "Classification Tree (rule = TRUE)","Random Forest","ANN MLP"), 
             col= c("dodgerblue3","#4daf4a","green", "orange", "red"), lwd=2)
    })#end of output$ML_most_accurate
    
    output$MLcomparison = renderPlot({

      #-----------------------------------------------------------------------------------------------------
      #### ROC & AUC Comparison between Models ####
      
      #KNN
      par(pty="s")
      roc(response = test_labels, predictor =  attr(wines_knn_pred[[input$knn]], "prob"), plot = T, 
          legacy.axes = T, percent = T, print.auc=T,
          xlab="% False Positive", ylab="% True Positive", 
          col="dodgerblue3",lwd = 2, print.auc.x=55, print.auc.y=65)
      
      #decision trees
      if(input$rule == FALSE){
        roc(response = test_labels, predictor = (predict(tree[[input$decision_trees]], test, type = "prob"))[,2] , plot = T, 
            percent = T, print.auc = T, add=T,
            col="#4daf4a", lwd=2, print.auc.x=55, print.auc.y=55)
      }#end of if statement
      else if(input$rule == TRUE){
        roc(response = test_labels, predictor = (predict(treeT[[input$decision_trees]], test, type = "prob"))[,2] , plot = T, 
            percent = T, print.auc = T, add=T,
            col="#4daf4a", lwd=2, print.auc.x=55, print.auc.y=55)
      }#END OF else if statement
      
      #random forest
      roc(response = test_labels, predictor = predict(rf[[input$random_forest]], test, type = "prob")[,2] , plot = T, 
          percent = T, print.auc = T, add=T,
          col="orange", lwd=2, print.auc.x=25, print.auc.y=65)

      #ANN MLPpredict
      roc(response = test_labels, predictor = predict(model.mlp[[as.numeric(input$ANN_MLP)]], test, type = "prob")[,2], plot = T, 
         percent = T, print.auc=T, add=T,
       col="red",lwd = 2, print.auc.x=25, print.auc.y=55)
      
      legend("bottomright",
             legend = c(paste("kNN (k = ",input$knn,")"),"Classification Tree", "Random Forest","ANN MLP"), 
             col= c("dodgerblue3","#4daf4a","orange", "red"), lwd=2)
      
    })#end of output$MLcomparison
   
   ########## START OF MODELS COMPARISONS #####
   output$knn = renderPlot({
     # ROC Curve (Receiver Operating Characteristic) & AUC
     par(pty="s")
     roc(response = test_labels, predictor =  knn_max_acc_prob, main="kNN with variable k values",
         plot=T,legacy.axes=T,percent=T,print.auc=T,
         xlab="% False Positive", ylab="% True Positive", col="dodgerblue3",lwd = 2)
     roc(response = test_labels, predictor = knn_FP_prob, plot = T, percent = T, print.auc = T,
         col="#4daf4a", lwd=2, add=T, print.auc.y=40)
     roc(response = test_labels, predictor = knn_FN_prob , plot = T, percent = T, print.auc = T,
         col="orange", lwd=2, add=T, print.auc.y=30)
     legend("bottomright", legend = c(paste("Max Accuracy (k = ",knn_max_acc_i,")"),
                                      paste("Min False Positive (k = ", knn_min_FP_i,")"),
                                      paste("Min False Negative (k = ", knn_min_FN_i,")")),
            col= c("dodgerblue3","#4daf4a", "orange"), lwd=2)
   })#end of output$knn
   
   output$random_forest = renderPlot({
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
   })#end of output$random_forest
   
   output$decision_trees = renderPlot({
     # ROC Curve (Receiver Operating Characteristic) & AUC
     par(pty="s")
     roc(response = test_labels, predictor = tree_max_acc , main=" Decision Trees with  variable Trials",
         plot=T,legacy.axes=T,percent=T,print.auc=T, 
         xlab="% False Positive", ylab="% True Positive", col="dodgerblue3",lwd = 2)
     roc(response = test_labels, predictor = tree_min_FP, plot = T, percent = T, print.auc = T,
         col="#4daf4a", lwd=2, add=T, print.auc.y=40)
     roc(response = test_labels, predictor = tree_min_FN , plot = T, percent = T, print.auc = T,
         col="orange", lwd=2, add=T, print.auc.y=30)
     legend("bottomright", legend = c(paste("Max Accuracy (n trees = ",tree_max_acc_i,")"),
                                      paste("Min False Positive (n trees = ",tree_min_FP_i,")"),
                                      paste("Min False Negative (n trees = ",tree_min_FN_i,")")), 
            col= c("dodgerblue3","#4daf4a", "orange"), lwd=2)
   })#end of output$decision_trees
   ########## END OF MODELS COMPARISONS #####
   
   output$ANN_MLP = renderPlot({})#end of output$ANN_MLP
   
   ####****#### TABS FUNCTIONS ####****####
   
           ### DATA ANALYSIS ###
           output$data_analysis_graphs = renderPlot({
        
             if(input$data_visualization == 1){
               #### Correlation Matrix ####
               # Check correlation for multicollinearity 
               corrmat <- as.matrix(cor(wines[,1:12]))
               ggcorrplot(corrmat)
             }#end of else if statement
             else if(input$data_visualization == 2){
               ####Boxplot of the variables####
               boxplot(wines_n [-12],horizontal=FALSE,axes=TRUE,outline=FALSE,col=(c("gold","darkgreen")),
                       main="Boxplot of variables")
             }#end of else if statement
           })#end of output$data_analysis
   
          output$data_analysis_legend = renderText({
            if(input$data_visualization == 1){
              (legend = "1")
            }#end of if statement
            else if(input$data_visualization == 2){
              (legend = "2")
            }#end of else if statement
          })#end of output$data_analysis_legend
   
           output$Demonstration <- renderUI({
     sidebarLayout(
       sidebarPanel(
         #########################################TEST DIFFERENT MODELS
         h3("Test different models"),
         p("Machine Learning models learn. But you can tell them how much to lear."),
         strong("Here, you can try out different variations of our models."),
         p(""),
         
         
         em("KNN"),
         sliderInput("knn",
                     "Number of groups",
                     step = 1,
                     min = 1,
                     max = 80,
                     value = 1),
         
         em("Decision Trees"),
         sliderInput("decision_trees",
                     "Number of trees",
                     step = 1,
                     min = 1,
                     max = 40,
                     value = 1),
         radioButtons("rule",
                      "RULE or NO RULE",
                      inline = TRUE,
                      choiceNames = c("RULE", "NO RULE"),
                      choiceValues = c(TRUE, FALSE)),
         
         em("Random Forest"),
         sliderInput("random_forest",
                     "Number of random variables",
                     step = 1,
                     min = 1,
                     max = 10,
                     value = 1),
         
         em("ANN MLP"),
         radioButtons("ANN_MLP",
                      "Number of neurons",
                      inline = TRUE,
                      choices = c("1" = 1,"2" = 2, "3" = 3))
       ),#end of sideBar Panel
       
       mainPanel(
         h2("Data Analysis"),
         
         p("A ROC curve is constructed by plotting the true positive rate (TPR) against the false positive rate (FPR). 
           The true positive rate is the proportion of observations that were correctly predicted to be positive out of 
           all positive observations (TP/(TP + FN)). Similarly, the false positive rate is the proportion of observations 
           that are incorrectly predicted to be positive out of all negative observations (FP/(TN + FP))."),
         
         p("The ROC curve shows the trade-off between sensitivity (or TPR) and specificity (1 – FPR). Classifiers that give
           curves closer to the top-left corner indicate a better performance. As a baseline, a random classifier is expected 
           to give points lying along the diagonal (FPR = TPR). The closer the curve comes to the 45-degree diagonal of the ROC
           space, the less accurate the test."),
         
         p("AUC: Area Under the Curve"),
         
         strong("Twiggle the buttons to see how iterations have an impact on the final graph"),
         plotOutput("MLcomparison"),
         
         #plotOutput("mytable_comparison"),
         plotOutput("mytable"),
         
         
         h2("Comparison bewteen models"),
         h4("Most accurate models"),
         p("The graph below shows the most accurate models."),
         plotOutput("ML_most_accurate"),
         
         h4("kNN"),
         p("The graph below shows different variations of kNN."),
         plotOutput("knn"),
         
         h4("Decision trees"),
         p("The graph below shows different variations of a decision tree."),
         plotOutput("decision_trees"),
         
         h4("Random Forest"),
         p("The graph below shows different variations of Random Forest"),
         plotOutput("random_forest")
         
         #h4("ANN MLP"),
         #p("The graph below shows different variations of ANN MLP"),
         #plotOutput("ANN")
       )# end of main panel
     )# end of sidebar layout
   })#end of output$Demonstration
           
           output$Data_Analysis = renderUI({
             sidebarLayout(
               sidebarPanel(
                 radioButtons("data_visualization",
                              "choose data visualizaton",
                              inline = FALSE,
                              choices = c("Correlations matrix between variables" = 1,
                                          "Boxplot of variables" = 2))
               ),#end of sideBar Panel
               mainPanel(
                 plotOutput("data_analysis_graphs"),
                 textOutput("data_analysis_legend")
               )#end of main
             )#end of sidebarLayout
           })#end of Data_Analysis
           ### END OF DATA ANALYSIS ###
           
           output$Model_Explanation <- renderUI({
             mainPanel(
               HTML(markdown::markdownToHTML(knit('ML Model Explanation.rmd', quiet = TRUE)))
             )#end of mainPanel
           })# end of output$Model_Explanation
           
           output$mytable = renderPlot({

             #### model comparison
             
             results <- matrix(NA, nrow = 4, ncol = 5)
             colnames(results) <- c("kNN","Tree (r = F)", "Tree (r = T)","RForest","ANN MLP")
             rownames(results) <- c("Type I error","Type II error",
                                    "True Positive","True Negative")
             results
             
             (knn_comp <- knn_confmatrix[[input$knn]])
             (tree_comp <- tree_confmatrix[[input$decision_trees]])
             (treeT_comp <- treeT_confmatrix[[input$decision_trees]])
             (rf_comp <- rf_confmatrix[[input$random_forest]])
             
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
             results[1,3] <- treeT_comp$table[1,2]
             results[2,3] <- treeT_comp$table[2,1]
             results[3,3] <- treeT_comp$table[1,1]
             results[4,3] <- treeT_comp$table[2,1]
             results
             results[1,4] <- rf_comp$table[1,2]
             results[2,4] <- rf_comp$table[2,1]
             results[3,4] <- rf_comp$table[1,1]
             results[4,4] <- rf_comp$table[2,1]
             results
             results[1,5] <- ann_co$table[1,2]
             results[2,5] <- ann_comp$table[2,1]
             results[3,5] <- ann_comp$table[1,1]
             results[4,5] <- ann_comp$table[2,1]
             results
             
             gc = tableGrob(results)
             
             ### statistics
             
             # Statistics of Models
             results_s <- matrix(NA, nrow = 3, ncol = 5)
             colnames(results_s) <- c("kNN","Tree (r = F)", "Tree (r = T)","RForest","ANN MLP")
             rownames(results_s) <- c("Accuracy","Sensitivity","Specificity")
             
             results_s[1,1] <- round(knn_comp$overall[1],3)
             results_s[2,1] <- round((knn_comp$table[1,1]/(knn_comp$table[1,1]+knn_comp$table[2,1])),3)
             results_s[3,1] <- round((knn_comp$table[2,2]/(knn_comp$table[2,2]+knn_comp$table[1,2])),3)
             
             results_s[1,2] <- round(treeT_comp$overall[1],3)
             results_s[2,2] <- round((treeT_comp$table[1,1]/(treeT_comp$table[1,1]+treeT_comp$table[2,1])),3)
             results_s[3,2] <- round((treeT_comp$table[2,2]/(treeT_comp$table[2,2]+treeT_comp$table[1,2])),3)
             
             results_s[1,3] <- round(tree_comp$overall[1],3)
             results_s[2,3] <- round((tree_comp$table[1,1]/(tree_comp$table[1,1]+tree_comp$table[2,1])),3)
             results_s[3,3] <- round((tree_comp$table[2,2]/(tree_comp$table[2,2]+tree_comp$table[1,2])),3)
             
             results_s[1,4] <- round(rf_comp$overall[1],3)
             results_s[2,4] <- round((rf_comp$table[1,1]/(rf_comp$table[1,1]+rf_comp$table[2,1])),3)
             results_s[3,4] <- round((rf_comp$table[2,2]/(rf_comp$table[2,2]+rf_comp$table[1,2])),3)
             
             results_s[1,5] <- round(ann_comp$overall[1],3)
             results_s[2,5] <- round((ann_comp$table[1,1]/(ann_comp$table[1,1]+ann_comp$table[2,1])),3)
             results_s[3,5] <- round((ann_comp$table[2,2]/(ann_comp$table[2,2]+ann_comp$table[1,2])),3)
             results_s
             
             theme = ttheme_default(base_size = 18, base_colour = "black", base_family = "",
                                    parse = FALSE)
             
             gs = tableGrob(results_s)

             gc$widths <- unit(rep(1/ncol(gc), ncol(gc)), "npc")
             gs$widths <- unit(rep(1/ncol(gs), ncol(gs)), "npc")
             grid.arrange(gc,gs, ncol=1, newpage = FALSE)

           })#end of output$mytable
   
   ####****#### END OF TABS FUNCTIONS ####****####
}#end of server
# Run the application 
shinyApp(ui = ui, server = server)