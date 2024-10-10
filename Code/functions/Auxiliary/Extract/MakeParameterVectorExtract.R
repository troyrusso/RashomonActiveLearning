### Summary: 
### Inputs:
### Output:

### Set Up ###
library(tidyverse)
rm(list=ls())
dir = paste0("/Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/")

ExpandGridCombinations = expand.grid(RashomonModelNumLimit = c(10,25,100))

### Run SLURM Simulations ###
Simulation_Combinations = ExpandGridCombinations %>%
  mutate(JobName = paste0("Extract", "_",
                          RashomonModelNumLimit),
         Output = paste0("Results/Extracted",
                         RashomonModelNumLimit,
                         ".RData"))
write.csv(Simulation_Combinations, file = paste0(dir,"ExtractedResults.csv"))





