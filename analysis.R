# Source data from Kaggle directly so that it's reproducible for the instructors without having to download the data themselves

library(data.table)
library(corrplot)
library(dplyr)
library(ggplot2)
library(magrittr)


properties_2016<-read.csv("properties_2016.csv",stringsAsFactors = F)
# This takes a while...~3 million rows, 58 columns
str(properties_2016)
n=nrow(properties_2016)

missingness<-lapply(properties_2016,function(x){
    ifelse(is.character(x),
          sum(identical(x,character(0))),
          sum(is.na(x))
  )
})

missingness<-round(unlist(missingness)/n*100,2)
missingness<-sort(missingness,decreasing=T)
missingness<-as.data.frame(missingness)
missingness$var<-row.names(missingness)
row.names(missingness)<-seq(1:nrow(missingness))
missingness<-select(missingness,c("var","missingness"))

ggplot(missingness,aes(x=reorder(var, -missingness),y=missingness)) +
  geom_col()

barplot(as.matrix(m))

M <- cor(properties_2016)

# Handle categorical variables
# Going to need to handle missing data - either throw it out or impute
  # Create a colume for "imputed"
# Use linear model as the baseline - whatever your machine learning algorithm is should do better than a linear model
# Predict by the mean as a baseline too - if the machine learning algorithm does worse than predicting by the mean, it's incorrect

train<-read.csv("train_2016_v2.csv")