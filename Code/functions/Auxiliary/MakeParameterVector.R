### Summary: 
### Inputs:
### Output:

### Set Up ###
library(tidyverse)
rm(list=ls())
dir = paste0("/Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/")

# seed = args$seed
# N = args$N
# K = args$K
# NClass = args$NClass
# # ClassProportion = args$ClassProportion
# CovCorrVal = args$CovCorrVal
# NBins = args$NBins
# ModelType = args$ModelType
# SelectorType = args$SelectorType
# TestProportion = args$TestProportion
# SelectorN = args$SelectorN
# InitialN = args$InitialN
# reg = args$reg
# theta = args$theta
# LabelName = args$LabelName
# Output = args$Output


ExpandGridCombinations = expand.grid(seed = seq(1:20),
                                     ModelType = c("Factorial", "RashomonLinear"),
                                     SelectorType = c("Random", "BreakingTies"),
                                     # N = c(1000),
                                     # N = c(5000),
                                     N = c(100),
                                     K = c(4),
                                     NClass = c(2),
                                     # ClassProportion = c(NA),
                                     CovCorrVal = c(0),
                                     NBins = c(3),
                                     TestProportion = c(0.2),
                                     SelectorN = c(1),
                                     InitialN = c(10),
                                     reg = c(0.1),
                                     theta = 2,
                                     LabelName = "YStar")

### Run SLURM Simulations ###
Simulation_Combinations = ExpandGridCombinations %>%
  mutate(JobName = paste0("Sim", "_",
                          seed, "_",
                          ModelType, "_",
                          SelectorType, "_",
                          N, "_",
                          K, "_",
                          NClass, "_",
                          # ClassProportion, "_",
                          CovCorrVal, "_",
                          NBins, "_",
                          TestProportion, "_",
                          SelectorN, "_",
                          InitialN,
                          reg, "_",
                          theta, "_",
                          LabelName),
         Output = paste0("Results/",
                         seed, "_",
                         ModelType, "_",
                         SelectorType, "_",
                         N, "_",
                         K, "_",
                         NClass, "_",
                         # ClassProportion, "_",
                         CovCorrVal, "_",
                         NBins, "_",
                         TestProportion, "_",
                         SelectorN, "_",
                         InitialN, 
                         ".RData"))
write.csv(Simulation_Combinations, file = paste0(dir,"ParameterVectorALL.csv"))





