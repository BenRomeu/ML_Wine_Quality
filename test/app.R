#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(DT)


# Define UI for application that draws a histogram
ui <- fluidPage(
  
  DT::dataTableOutput("mytable")
  
   
  )

# Define server logic required to draw a histogram
server <- function(input, output) {
   
  output$mytable = DT::renderDataTable({
    
    (knn_comp <- knn_confmatrix[[5]])
    (tree_comp <- tree_confmatrix[[5]])
    (rf_comp <- rf_confmatrix[[5]])
    
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
  })#end of output$mytable
}

# Run the application 
shinyApp(ui = ui, server = server)

