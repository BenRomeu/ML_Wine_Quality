####Data Visualization####


#### Correlation Matrix ####

# Check correlation for multicollinearity 
corrmat <- as.matrix(cor(wines[,1:12]))
ggcorrplot(corrmat)

####Boxplot of the variables####

boxplot(wines_n[-12])


#####Distribution of Good and Bad quality with 2 variables(alcohol and chlorides)####

ggplot(aes(x=alcohol,y=chlorides, colour= label), data = wines_f) +
 stat_smooth(method=loess, fullrange=TRUE, alpha = 0.1, size =1.5 )


