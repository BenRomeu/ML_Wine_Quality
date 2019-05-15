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
library(shinyjs)

  #### Load database of red wines and white wines ####
  load(paste(getwd(),"/data_set.RData", sep = ""), envir = parent.frame(), verbose = FALSE)

#-----------------------------------------------------------------------------------------------------
#### Shiny App

# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("Machine Learning Algorithms - How To Predict Wine Quality"),
   
   tabsetPanel(id = "tabSelected",
               tabPanel("Demonstration", uiOutput("Demonstration")),
               tabPanel("Model Explanation", uiOutput("Model_Explanation"))
   )#end of tabsetPanel
   
)# end of ui fluidpage

# Define server logic required to draw a histogram
server <- function(input, output) {
  
### DATA ANALYSIS ###
    output$data_analysis = renderTable({#TODO: replace with actual data analysis from Donika
      
      wines = input$dataSetChoice[,1]
      print(wines)
      
      # Histogram of each features
      par(mfrow=c(2,3))
      for(i in 1:6) {
        hist(wines[,i], main=names(wines)[i])
      }
      for(i in 7:12) {
        hist(wines[,i], main=names(wines)[i])
      }
        #plot(input$dataSetChoice, ylim=c(0,100), xlim = c(0,100))
    })#end of output$data_analysis

    
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
      roc(response = test_labels, predictor = (predict(tree[[input$decision_trees]], test, type = "prob"))[,2] , plot = T, 
         percent = T, print.auc = T, add=T,
        col="#4daf4a", lwd=2, print.auc.x=55, print.auc.y=55)
      
      #roc(response = test_labels, predictor = rpart_prob , plot = T, 
       #   percent = T, print.auc = T, add=T,
        #  col="purple", lwd=2, print.auc.x=55, print.auc.y=45)
      
      #random forest
      roc(response = test_labels, predictor = predict(rf[[input$random_forest]], test, type = "prob")[,2] , plot = T, 
          percent = T, print.auc = T, add=T,
          col="orange", lwd=2, print.auc.x=25, print.auc.y=65)

      #ANN MLPpredict
      
      roc(response = test_labels, predictor = predict(model.mlp[[as.numeric(input$ANN_MLP)]], test, type = "prob")[,2], plot = T, 
         percent = T, print.auc=T, add=T,
       col="red",lwd = 2, print.auc.x=25, print.auc.y=55)
      
      legend("bottomright",
             legend = c(paste("kNN (k = ",input$knn),"Classification Tree","RPART Tree", "Random Forest","ANN MLP"), 
             col= c("dodgerblue3","#4daf4a","purple","orange", "red"), lwd=2)
      
    })#end of output$MLcomparison
    
    output$test = renderPlot({
      plot(input$ANN_MLP)
    })
     
   output$MLPplot <- renderPlot({
      # generate bins based on input$bins from ui.R
     
     
     # ANN MLP Model
     model.mlp = neuralnet(label ~.,data = wines_n, hidden = 1, linear.output = FALSE)
     
      #alcohol <- seq(min(x), max(x), length.out = input$alcohol+1)
      
      # plot NN with the specified amount of alcohol
      plot.nnet(model.mlp)
   })#end of output$MLPplot
   
   output$my_wine_prediction = renderText({
     
     #assign user alcohol value (11th row) to first test element
     test[1,11] = input$alcohol
     model.mlp_prob <- predict(model.mlp, test[1,], type ="prob")[,2]
     label = ifelse(model.mlp_prob>0.5,"Good", "Bad")
     print(label)
     
   })#end of output$my_wine_prediction
   
   output$decision_trees = renderPlot({})#end of output$decision_trees
   
   output$rule = renderPlot({})#end of output$rule
   
   output$knn = renderPlot({})#end of output$knn
   
   output$random_forest = renderPlot({})#end of output$random_forest
   
   output$RPART = renderPlot({})#end of output$RPART
   
   output$ANN_MLP = renderPlot({})#end of output$ANN_MLP
   
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
           p("This graph compares the efficiency of different models"),
           plotOutput("MLcomparison")
         )# end of main panel
       )# end of sidebar layout
   })#end of output$Demonstration
   
   output$Model_Explanation <- renderUI({
     mainPanel(
       #uiOutput('markdown')
       HTML(markdown::markdownToHTML(knit('ML Model Explanation.rmd', quiet = TRUE)))
     )#end of mainPanel
   })# end of output$Model_Explanation
   
}#end of server
# Run the application 
shinyApp(ui = ui, server = server)