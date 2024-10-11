### Summary:
### Inputs:
### Output:

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
  make_option(c("--ModelType"), type = "character", default = "Logistic", help = "Predictor model type", metavar = "character"),
  make_option(c("--SelectorType"), type = "character", default = "Random", help = "Selector type", metavar = "character"),
  make_option(c("--TestProportion"), type = "numeric", default = 0.2, help = "Test set proportion", metavar = "numeric"),
  make_option(c("--SelectorN"), type = "numeric", default = 1, help = "Number of observations to query", metavar = "numeric"),
  make_option(c("--InitialN"), type = "numeric", default = 10, help = "Initial number of classes", metavar = "numeric"),
  make_option(c("--reg"), type = "numeric", default = 0.1, help = "Penalty on the splits", metavar = "numeric"),
  make_option(c("--theta"), type = "numeric", default = 3, help = "Rashomon Threshold", metavar = "numeric"),
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
theta = args$theta
RashomonModelNumLimit = args$RashomonModelNumLimit
output = args$output

seed = 1
ModelType = "RashomonLinear"
SelectorType = "BreakingTies"
TestProportion = 0.2
SelectorN = 1
InitialN = 10
reg = 0.1
theta = 1
RashomonModelNumLimit = 10

### Recall Data ###
 if(exists("directory")){
   dat = read.csv(paste0(directory,"Data/AmesHousingDataProcessed.csv"))
 }else if(!exists("directory")){dat = read.csv("Data/AmesHousingDataProcessed.csv")}
 LabelName = "SalePrice"
 R = c(11,4)
 CovariateList = c("OverallQuality",
                   "YearBuilt")
 
### Set Up ###
RashomonParameters = list(H = Inf,                           # Maximum number of pools/splits
                          R = R,                       # Bins of each arm (assume 0 exists)
                          reg = 0.1,                         # Penalty on the splits
                          theta = theta,                         # Threshold; determine relative to best model
                          inactive = 0,
                          RashomonModelNumLimit = RashomonModelNumLimit)

### Simulation ###
SimulationFunc(dat = dat[1:100,],
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






