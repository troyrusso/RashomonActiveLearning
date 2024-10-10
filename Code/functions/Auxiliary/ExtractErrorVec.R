### Directory ###
rm(list=ls())
directory = "/Users/simondn/Documents/RashomonActiveLearning/Results"
source("/Users/simondn/Documents/RashomonActiveLearning/Code/functions/Plot/SelectorTypeComparisonPlotFuncDynamic.R")

### Auxiliary Function ###
AddRowToMatrix = function(StorageMatrix, NewRow){
  if(ncol(StorageMatrix) == 0){StorageMatrix = matrix(NewRow, nrow = 1, ncol = length(NewRow))
  } else { StorageMatrix = rbind(StorageMatrix, NewRow)}
}

### Data ###
RDataFiles_FactorialRandom = list.files(path = directory, 
                                         pattern = "Factorial_Random.*\\.RData$", 
                                         full.names = TRUE)
RDataFiles_FactorialBreakingTies = list.files(path = directory, 
                                               pattern = "Factorial_BreakingTies.*\\.RData$", 
                                               full.names = TRUE)
RDataFiles_RashomonBreakingTies = list.files(path = directory, 
                                              pattern = "RashomonLinear_BreakingTies.*\\.RData$", 
                                              full.names = TRUE)

### 10 Rashomon Set ###
RDataFiles_FactorialRandom_Short = RDataFiles_FactorialRandom[!grepl("25_\\.RData$", 
                                                                     RDataFiles_FactorialRandom)]
RDataFiles_FactorialBreakingTies_Short = RDataFiles_FactorialBreakingTies[!grepl("25_\\.RData$", 
                                                                                 RDataFiles_FactorialBreakingTies)]
RDataFiles_RashomonBreakingTies_Short = RDataFiles_RashomonBreakingTies[!grepl("25_\\.RData$", 
                                                                               RDataFiles_RashomonBreakingTies)]

### 25 Rashomon Set ###
# RDataFiles_FactorialRandom_Long = RDataFiles_FactorialRandom[grepl("25_\\.RData$", 
#                                                                      RDataFiles_FactorialRandom)]
# RDataFiles_FactorialBreakingTies_Long = RDataFiles_FactorialBreakingTies[grepl("25_\\.RData$", 
#                                                                                  RDataFiles_FactorialBreakingTies)]
# RDataFiles_RashomonBreakingTies_Long = RDataFiles_RashomonBreakingTies[grepl("25_\\.RData$", 
                                                                               # RDataFiles_RashomonBreakingTies)]

### CHOOSE ###
RDataFiles_FactorialRandom_DO = RDataFiles_FactorialRandom_Short
RDataFiles_FactorialBreakingTies_DO = RDataFiles_FactorialBreakingTies_Short
RDataFiles_RashomonBreakingTies_DO = RDataFiles_RashomonBreakingTies_Short

### Validation ###
LengthVar = var(c(length(RDataFiles_FactorialRandom_DO), 
                  length(RDataFiles_FactorialBreakingTies_DO), 
                  length(RDataFiles_RashomonBreakingTies_DO)))
if(LengthVar!=0){warning("Result lenghts are not all the same.")}

### Set Up ###

## List ##
SimulationResultsList_FactorialRandom = list()
SimulationResultsList_FactorialBreakingTies = list()
SimulationResultsList_RashomonBreakingTies = list()

## Error Vec ##
ErrorVec_FactorialRandom = matrix(ncol= 0, nrow = 0)
ErrorVec_FactorialBreakingTies = matrix(ncol= 0, nrow = 0)
ErrorVec_RashomonBreakingTies = matrix(ncol = 0, nrow = 0)

### Progress Bar ###
pb = txtProgressBar(min = 0, 
                    max = length(RDataFiles_FactorialRandom_DO),
                    style = 3,  
                    width = 50,
                    char = "=")
### Loop ###
for (i in 1:length(RDataFiles_FactorialRandom_DO)) {
  
  ## Progress Bar ##
  setTxtProgressBar(pb, i)
  print(i)

  ## Random ##
  load(RDataFiles_FactorialRandom_DO[i])
  SimulationResultsList_FactorialRandom[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec_FactorialRandom = AddRowToMatrix(ErrorVec_FactorialRandom,SimulationResults$ErrorVec)
  rm(SimulationResults)
  
  ## Factorial ##
  load(RDataFiles_FactorialBreakingTies_DO[i])
  SimulationResultsList_FactorialBreakingTies[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec_FactorialBreakingTies = AddRowToMatrix(ErrorVec_FactorialBreakingTies,SimulationResults$ErrorVec)
  rm(SimulationResults)
  
  ## Rashomon ##
  load(RDataFiles_RashomonBreakingTies_DO[i])
  SimulationResultsList_RashomonBreakingTies[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec_RashomonBreakingTies = AddRowToMatrix(ErrorVec_RashomonBreakingTies,SimulationResults$ErrorVec)
  rm(SimulationResults)
  }

### Get Mean ###
MeanErrorVec_FactorialRandom = colMeans(ErrorVec_FactorialRandom)
MeanErrorVec_FactorialBreakingTies = colMeans(ErrorVec_FactorialBreakingTies)
MeanErrorVec_RashomonBreakingTies = colMeans(ErrorVec_RashomonBreakingTies)

plot(MeanErrorVec_FactorialRandom)
plot(MeanErrorVec_FactorialBreakingTies)
plot(MeanErrorVec_RashomonBreakingTies)

Plot2 <- SelectorTypeComparisonPlotFuncDynamic(
  # Random = MeanErrorVec_FactorialRandom,
  Naive = MeanErrorVec_FactorialBreakingTies,
  `Rashomon-Weighted` = MeanErrorVec_RashomonBreakingTies,
  xlower =0,
  xupper = 230
)

Plot3 <- SelectorTypeComparisonPlotFuncDynamic(
  Random = MeanErrorVec_FactorialRandom,
  Naive = MeanErrorVec_FactorialBreakingTies,
  `Rashomon-Weighted` = MeanErrorVec_RashomonBreakingTies,
  xlower = 0,
  xupper = 230
)

Plot2
Plot3





