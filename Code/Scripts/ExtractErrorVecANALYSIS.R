### Set Up ###
rm(list=ls())
directory = "/Users/simondn/Documents/RashomonActiveLearning/"
library(MASS) 
library(tidyverse)
library(ggplot2) 
library(dplyr)
library(class)
library(glmnet)
library(nnet)
library(RcppAlgos)   #TVA
library(data.table)  #TVA
library(rashomontva) #TVA

if(exists("directory")){
  source(paste0(directory,"Code/functions/Auxiliary/LoadFunctions.R"))
}else if(!exists("directory")){source("Code/functions/Main/LoadFunctions.R")}

# save(OutputVector, file= "/Users/simondn/Documents/RashomonActiveLearning/Results/PreliminarResults.RData")
load("/Users/simondn/Documents/RashomonActiveLearning/Results/PreliminarResults.RData")


SelectorTypeComparisonPlotFunc(OutputVector$MeanOutputVector)

