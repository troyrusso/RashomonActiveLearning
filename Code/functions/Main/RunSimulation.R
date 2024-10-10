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
  make_option(c("--ModelType"), type = "character", default = "Logistic", help = "Predictor model type", metavar = "character"),
  make_option(c("--SelectorType"), type = "character", default = "Random", help = "Selector type", metavar = "character"),
  make_option(c("--N"), type = "integer", default = 100, help = "Number of observations", metavar = "integer"),
  make_option(c("--K"), type = "integer", default = 4, help = "Number of covariates", metavar = "integer"),
  make_option(c("--NClass"), type = "integer", default = 2, help = "Number of classes", metavar = "integer"),
  # make_option(c("--ClassProportion"), type = "complex", default = NULL, help = "Proportion of classes", metavar = "complex"),
  make_option(c("--CovCorrVal"), type = "numeric", default = -0.4, help = "Correlation between covariate 1 and 2", metavar = "numeric"),
  make_option(c("--NBins"), type = "numeric", default = 3, help = "Discretizes data into NBins.", metavar = "numeric"),
  make_option(c("--TestProportion"), type = "numeric", default = 0.2, help = "Test set proportion", metavar = "numeric"),
  make_option(c("--SelectorN"), type = "numeric", default = 1, help = "Number of observations to query", metavar = "numeric"),
  make_option(c("--InitialN"), type = "numeric", default = 1, help = "Initial number of classes", metavar = "numeric"),
  make_option(c("--reg"), type = "numeric", default = 0.1, help = "Penalty on the splits", metavar = "numeric"),
  make_option(c("--theta"), type = "numeric", default = 3, help = "Rashomon Threshold", metavar = "numeric"),
  make_option(c("--LabelName"), type = "character", default = "Y", help = "Y or YStar", metavar = "character"),
  make_option(c("--output"), type = "character", default = NULL, help = "Path to store", metavar = "character")
)
arg.parser = OptionParser(option_list = option_list)
args = parse_args(arg.parser)

## Parameters ##
seed = args$seed
ModelType = args$ModelType
SelectorType = args$SelectorType
N = args$N
K = args$K
NClass = args$NClass
CovCorrVal = args$CovCorrVal
NBins = args$NBins
TestProportion = args$TestProportion
SelectorN = args$SelectorN
InitialN = args$InitialN
reg = args$reg
theta = args$theta
LabelName = args$LabelName
output = args$output

### Generate Data ###
set.seed(seed)
DGPResults = GenerateDataFunc(N, K, NClass, CovCorrVal, NBins = NBins)
dat = DGPResults$dat
TrueBetas = DGPResults$TrueBetas

### Set Up ###
CovariateList = paste0("X", 1:K)
ClassProportion = rep(1/NClass, NClass)
RashomonParameters = list(K = K, 
                          NBins = NBins,
                          H = Inf,                           # Maximum number of pools/splits
                          R = NBins+1,                       # Bins of each arm (assume 0 exists)
                          reg = 0.1,                         # Penalty on the splits
                          theta = 4,                         # Threshold; determine relative to best model
                          inactive = 0,
                          RashomonModelNumLimit = 10)

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
                                                  N = N,
                                                  K = K,
                                                  NClass = NClass,
                                                  ClassProportion = ClassProportion,
                                                  CovCorrVal = CovCorrVal,
                                                  TestProportion = TestProportion,
                                                  SelectorN = SelectorN,
                                                  InitialN = InitialN,
                                                  NBins = NBins,
                                                  RashomonParameters = RashomonParameters)
save(SimulationResults, file = output)






