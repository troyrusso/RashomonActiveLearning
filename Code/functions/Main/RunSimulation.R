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
  make_option(c("--job_name"), type = "character", default = "Bad", help = "Job Name", metavar = "integer"),
  make_option(c("--seed"), type = "integer", default = 69, help = "Seed", metavar = "integer"),
  make_option(c("--ModelType"), type = "character", default = "Bad", help = "Predictor model type", metavar = "character"),
  make_option(c("--SelectorType"), type = "character", default = "Bad", help = "Selector type", metavar = "character"),
  make_option(c("--N"), type = "integer", default = 69, help = "Number of observations", metavar = "integer"),
  make_option(c("--K"), type = "integer", default = 69, help = "Number of covariates", metavar = "integer"),
  make_option(c("--NClass"), type = "integer", default = 69, help = "Number of classes", metavar = "integer"),
  make_option(c("--CovCorrVal"), type = "numeric", default = 69, help = "Correlation between covariate 1 and 2", metavar = "numeric"),
  make_option(c("--NBins"), type = "numeric", default = 69, help = "Discretizes data into NBins.", metavar = "numeric"),
  make_option(c("--TestProportion"), type = "numeric", default = 69, help = "Test set proportion", metavar = "numeric"),
  make_option(c("--SelectorN"), type = "numeric", default = 69, help = "Number of observations to query", metavar = "numeric"),
  make_option(c("--InitialN"), type = "numeric", default = 69, help = "Initial number of classes", metavar = "numeric"),
  make_option(c("--reg"), type = "numeric", default = 69, help = "Penalty on the splits", metavar = "numeric"),
  make_option(c("--theta"), type = "numeric", default = 69, help = "Rashomon Threshold", metavar = "numeric"),
  make_option(c("--RashomonModelNumLimit"), type = "numeric", default = 69, help = "Max Rashomon number", metavar = "numeric"),
  make_option(c("--LabelName"), type = "character", default = "Bad", help = "Y or YStar", metavar = "character"),
  make_option(c("--output"), type = "character", default = "Bad", help = "Path to store", metavar = "character")
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
RashomonModelNumLimit = args$RashomonModelNumLimit
LabelName = args$LabelName
output = args$output

if(seed == 69){print("seed is Bad")}
if(ModelType == "Bad"){print("ModelType is Bad")}
if(SelectorType == "Bad"){print("SelectorType is Bad")}
if(N == 69){print("N is Bad")}
if(K == 69){print("K is Bad")}
if(NClass == 69){print("NClass is Bad")}
if(CovCorrVal == 69){print("CovCorrVal is Bad")}
if(NBins == 69){print("NBins is Bad")}
if(TestProportion == 69){print("TestProportion is Bad")}
if(SelectorN == 69){print("SelectorN is Bad")}
if(InitialN == 69){print("InitialN is Bad")}
if(reg == 69){print("reg is Bad")}
if(theta == 69){print("theta is Bad")}
if(RashomonModelNumLimit == 69){print("RashomonModelNumLimit is Bad")}
if(LabelName == "Bad"){print("LabelName is Bad")}
if(output == "Bad"){print("output is Bad")}

### Generate Data ###
set.seed(seed)
DGPResults = GenerateDataFunc(N, K, CovCorrVal, NBins = NBins)
dat = DGPResults$dat
TrueBetas = DGPResults$TrueBetas

### Set Up ###
CovariateList = paste0("X", 1:K)
RashomonParameters = list(K = K, 
                          NBins = NBins,
                          H = Inf,                           # Maximum number of pools/splits
                          R = NBins+1,                       # Bins of each arm (assume 0 exists)
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
                                                  N = N,
                                                  K = K,
                                                  CovCorrVal = CovCorrVal,
                                                  TestProportion = TestProportion,
                                                  SelectorN = SelectorN,
                                                  InitialN = InitialN,
                                                  NBins = NBins,
                                                  RashomonParameters = RashomonParameters)
save(SimulationResults, file = output)






