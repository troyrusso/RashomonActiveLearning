### Summary: 
### Inputs:
### Output:

### Set Up ###
library(tidyverse)
rm(list=ls())
dir = paste0("/Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/")

ExpandGridCombinations = expand.grid(seed = seq(1:50),
                                     ModelType = c("Factorial", "RashomonLinear"),
                                     SelectorType = c("Random", "BreakingTies"),
                                     TestProportion = c(0.2),
                                     SelectorN = c(1),
                                     InitialN = c(10),
                                     reg = c(0.1),
                                     theta = 2,
                                     RashomonModelNumLimit = c(10, 25, 100),
                                     LabelName = "YStar")

### Filter out RashomonLinear_Random ###
ExpandGridCombinations <- ExpandGridCombinations[!(ExpandGridCombinations$ModelType == "RashomonLinear" & 
                                                       ExpandGridCombinations$SelectorType == "Random"), ]


### Run SLURM Simulations ###
Simulation_Combinations = ExpandGridCombinations %>%
  mutate(JobName = paste0("Sim", "_",
                          seed, "_",
                          ModelType, "_",
                          SelectorType, "_",
                          TestProportion, "_",
                          SelectorN, "_",
                          InitialN, "_",
                          reg, "_",
                          theta, "_",
                          RashomonModelNumLimit, "_",
                          LabelName),
         Output = paste0("Results/SimulationRaw",
                         seed, "_",
                         ModelType, "_",
                         SelectorType, "_",
                         TestProportion, "_",
                         SelectorN, "_",
                         InitialN, "_", 
                         reg, "_",
                         theta, "_",
                         RashomonModelNumLimit, "_",
                         ".RData"))
write.csv(Simulation_Combinations, file = paste0(dir,"ParameterVectorEMPIRICAL.csv"))





