### Directory ###
rm(list=ls())
directory = "/Users/simondn/Documents/RashomonActiveLearning/Results"

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

### Validation ###
LengthVar = var(c(length(RDataFiles_FactorialRandom), 
                  length(RDataFiles_FactorialBreakingTies), 
                  length(RDataFiles_RashomonBreakingTies)))
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
                    max = MaxIterationN,
                    style = 3,  
                    width = 50,
                    char = "=")
### Loop ###
for (i in 1:length(RDataFiles_FactorialRandom)) {
  
  ## Progress Bar ##
  setTxtProgressBar(pb, iter)

  ## Random ##
  load(RDataFiles_FactorialRandom[i])
  SimulationResultsList_FactorialRandom[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec_FactorialRandom = AddRowToMatrix(ErrorVec_FactorialRandom,SimulationResults$ErrorVec)
  rm(SimulationResults)
  
  ## Factorial ##
  load(RDataFiles_FactorialBreakingTies[i])
  SimulationResultsList_FactorialBreakingTies[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec_FactorialBreakingTies = AddRowToMatrix(ErrorVec_FactorialBreakingTies,SimulationResults$ErrorVec)
  rm(SimulationResults)
  
  ## Rashomon ##
  load(RDataFiles_RashomonBreakingTies[i])
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
  # MeanErrorVec_FactorialRandom = MeanErrorVec_FactorialRandom,
  MeanErrorVec_FactorialBreakingTies = MeanErrorVec_FactorialBreakingTies,
  MeanErrorVec_RashomonBreakingTies = MeanErrorVec_RashomonBreakingTies
)

Plot3 <- SelectorTypeComparisonPlotFuncDynamic(
  MeanErrorVec_FactorialRandom = MeanErrorVec_FactorialRandom,
  MeanErrorVec_FactorialBreakingTies = MeanErrorVec_FactorialBreakingTies,
  MeanErrorVec_RashomonBreakingTies = MeanErrorVec_RashomonBreakingTies
)

Plot3





