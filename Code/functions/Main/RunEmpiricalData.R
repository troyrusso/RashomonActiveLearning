### Summary:
### Inputs:
### Output:

### Library Directory ###
# .libPaths("/homes/simondn/Rlibs")
# .libPaths("~/Rlibs/x86_64-pc-linux-gnu/4.2.2")

# Set Up
## Libraries
library(MASS)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(class)
library(glmnet)
library(nnet)
library(data.table) #TVA
library(optparse)   #parse
library(RcppAlgos)  #TVA
library(rashomontva) #TVA

## Rm. Var.
rm(list=ls())

## My Functions
# directory = "/Users/simondn/Documents/RashomonActiveLearning/"

if(exists("directory")){
  source(paste0(directory,"Code/functions/Auxiliary/LoadFunctions.R"))
}else if(!exists("directory")){source("Code/functions/Auxiliary/LoadFunctions.R")}

## Parser ###
option_list = list(
  make_option(c("--job_name"), type = "character", default = NULL, help = "Job Name", metavar = "integer"),
  make_option(c("--seed"), type = "integer", default = 1, help = "seed", metavar = "integer"),
  make_option(c("--DataSetInput"), type = "character", default = NULL, help = "Data Set", metavar = "character"),
  make_option(c("--ModelType"), type = "character", default = "Logistic", help = "Predictor model type", metavar = "character"),
  make_option(c("--SelectorType"), type = "character", default = "Random", help = "Selector type", metavar = "character"),
  make_option(c("--TestProportion"), type = "numeric", default = 0.2, help = "Test set proportion", metavar = "numeric"),
  make_option(c("--SelectorN"), type = "numeric", default = 1, help = "Number of observations to query", metavar = "numeric"),
  make_option(c("--InitialN"), type = "numeric", default = 1, help = "Initial number of classes", metavar = "numeric"),
  make_option(c("--reg"), type = "numeric", default = 0.1, help = "Penalty on the splits", metavar = "numeric"),
  make_option(c("--RashomonModelNumLimit"), type = "numeric", default = 10, help = "Max Rashomon number", metavar = "numeric"),
  make_option(c("--output"), type = "character", default = NULL, help = "Path to store", metavar = "character")
)
arg.parser = OptionParser(option_list = option_list)
args = parse_args(arg.parser)

## Parameters ##
seed = args$seed
ModelType = args$ModelType
SelectorType = args$SelectorType
TestProportion = args$TestProportion
SelectorN = args$SelectorN
InitialN = args$InitialN
reg = args$reg
RashomonModelNumLimit = args$RashomonModelNumLimit
output = args$output

### Recall Data ###
switch(DataSetInput,
       GermanCredit = {
         if(exists("directory")){
           dat = read.csv(paste0(directory,"Data/GermanCreditDataProcessed.csv"))
         }else if(!exists("directory")){dat = read.csv("Data/GermanCreditDataProcessed.csv")}
         LabelName = "Credit"
         R = c(3,5,4,5)
         theta = 3
         CovariateList = c("Sex",
                           "Job",
                           "Housing",
                           "Savings")},
       AmesHousing = {
         if(exists("directory")){
           dat = read.csv(paste0(directory,"Data/DiamondsDataProcessed.csv"))
         }else if(!exists("directory")){dat = read.csv("Data/DiamondsDataProcessed.csv")}
         LabelName = "SalePrice"
         R = c(11,4,4)
         theta = 3
         CovariateList = c("OverallQuality",
                           "LivingArea",
                           "YearBuilt")
         },
         Diamonds = {
           if(exists("directory")){
             dat = read.csv(paste0(directory,"Data/AmesHousingDataProcessed.csv"))
             }else if(!exists("directory")){dat = read.csv("Data/AmesHousingDataProcessed.csv")}
       
           LabelName = "price"
           R = c(5,6,7)
           theta = 3
           CovariateList = c("carat",
                             "cut",
                             "color")}
         )

### Set Up ###
RashomonParameters = list(K = K, 
                          H = Inf,                           # Maximum number of pools/splits
                          R = R,                       # Bins of each arm (assume 0 exists)
                          reg = 0.1,                         # Penalty on the splits
                          theta = 4,                         # Threshold; determine relative to best model
                          inactive = 0,
                          RashomonModelNumLimit = RashomonModelNumLimit)

### Simulation ###
SimulationFunc(dat = dat,
               LabelName = LabelName,
               CovariateList = CovariateList,
               TestProportion = TestProportion,
               SelectorType = SelectorType,
               SelectorN = SelectorN,
               ModelType = ModelType,
               InitialN = InitialN,
               RashomonParameters = RashomonParameters,
               seed = seed) -> SimulationResults

### Save ###
SimulationResults$Parameters = list(seed = seed,
                                    dat = DataSetInput,
                                    TestProportion = TestProportion,
                                    SelectorN = SelectorN,
                                    InitialN = InitialN,
                                    NBins = NBins,
                                    RashomonParameters = RashomonParameters)
save(SimulationResults, file = output)






