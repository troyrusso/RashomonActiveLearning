### Summary: 
### Inputs:
### Output:

### Set Up ###
library(tidyverse)
rm(list=ls())
dir = paste0("/Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/")

ExpandGridCombinations = expand.grid(seed = seq(51,100),
                                     ModelType = c("Factorial", "RashomonLinear"),
                                     SelectorType = c("Random", "BreakingTies"),
                                     N = c(300),
                                     K = c(3),
                                     CovCorrVal = c(0),
                                     NBins = c(4),
                                     TestProportion = c(0.2),
                                     SelectorN = c(1),
                                     InitialN = c(10),
                                     reg = c(0.1),
                                     theta = 2,
                                     RashomonModelNumLimit = c(10, 25, 100),
                                     LabelName = "Y")

### Delete Extra Sbatch ###
ExpandGridCombinations = ExpandGridCombinations %>%
  filter(!(ModelType == "RashomonLinear" & SelectorType == "Random")) %>%  # Case 1
  filter(!(ModelType == "Factorial" & RashomonModelNumLimit != 10)) %>%    # Case 2
  filter(!(SelectorType == "Random" & RashomonModelNumLimit != 10))          # Case 3

### Run SLURM Simulations ###
Simulation_Combinations = ExpandGridCombinations %>%
  mutate(JobName = paste0("Sim", "_",
                          seed, "_",
                          ModelType, "_",
                          SelectorType, "_",
                          N, "_",
                          K, "_",
                          CovCorrVal, "_",
                          NBins, "_",
                          TestProportion, "_",
                          SelectorN, "_",
                          InitialN, "_",
                          reg, "_",
                          theta, "_",
                          RashomonModelNumLimit, "_",
                          LabelName),
         Output = paste0("Results/SimulationRaw/",
                         seed, "_",
                         ModelType, "_",
                         SelectorType, "_",
                         N, "_",
                         K, "_",
                         CovCorrVal, "_",
                         NBins, "_",
                         TestProportion, "_",
                         SelectorN, "_",
                         InitialN, "_", 
                         reg, "_",
                         theta, "_",
                         RashomonModelNumLimit, "_",
                         ".RData"))
write.csv(Simulation_Combinations, file = paste0(dir,"ParameterVectorALL.csv"))





