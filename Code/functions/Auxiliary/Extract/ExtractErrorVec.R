### Set Up ###
rm(list=ls())
directory = "Results/SimulationRaw"
directory = "/Users/simondn/Documents/RashomonActiveLearning/Results/SimulationRaw/"
library(optparse)   #parse

## Parser ###
option_list = list(
  make_option(c("--JobName"), type = "character", default = NULL, help = "Job Name", metavar = "integer"),
  make_option(c("--RashomonModelNumLimit"), type = "numeric", default = NULL, help = "Max Rashomon number", metavar = "numeric"),
  make_option(c("--output"), type = "character", default = NULL, help = "Path to store", metavar = "character")
)
arg.parser = OptionParser(option_list = option_list)
args = parse_args(arg.parser)
RashomonModelNumLimit = args$RashomonModelNumLimit
RashomonModelNumLimit = 10
output = args$output

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

RDataFiles_FactorialRandom = RDataFiles_FactorialRandom[grepl(paste0(RashomonModelNumLimit,"_\\.RData$"), 
                                                                     RDataFiles_FactorialRandom)]
RDataFiles_FactorialBreakingTies = RDataFiles_FactorialBreakingTies[grepl(paste0(RashomonModelNumLimit,"_\\.RData$"), 
                                                                                 RDataFiles_FactorialBreakingTies)]
RDataFiles_RashomonBreakingTies = RDataFiles_RashomonBreakingTies[grepl(paste0(RashomonModelNumLimit,"_\\.RData$"), 
                                                                               RDataFiles_RashomonBreakingTies)]

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

### Run Times ###
RunTimeRandom = c()
RunTimeFactorial = c()
RunTimeRashomon = c()
### Progress Bar ###
pb = txtProgressBar(min = 0, 
                    max = 50,
                    style = 3,  
                    width = 50,
                    char = "=")
### Loop ###
for (i in 1:50) {

  ## Progress Bar ##
  setTxtProgressBar(pb, i)
  print(i)
  
  # ## Random ##
  # load(RDataFiles_FactorialRandom[i])
  # SimulationResultsList_FactorialRandom[[paste0("SimulationResults_", i)]] = SimulationResults
  # ErrorVec_FactorialRandom = AddRowToMatrix(ErrorVec_FactorialRandom,SimulationResults$ErrorVec)
  # RunTimeRandom = c(RunTimeRandom, SimulationResults$run_time)
  # rm(SimulationResults)
  # 
  # ## Factorial ##
  # load(RDataFiles_FactorialBreakingTies[i])
  # SimulationResultsList_FactorialBreakingTies[[paste0("SimulationResults_", i)]] = SimulationResults
  # ErrorVec_FactorialBreakingTies = AddRowToMatrix(ErrorVec_FactorialBreakingTies,SimulationResults$ErrorVec)
  # RunTimeFactorial = c(RunTimeFactorial, SimulationResults$run_time)
  # rm(SimulationResults)
  
  ## Rashomon ##
  load(RDataFiles_RashomonBreakingTies[i])
  SimulationResultsList_RashomonBreakingTies[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec_RashomonBreakingTies = AddRowToMatrix(ErrorVec_RashomonBreakingTies,SimulationResults$ErrorVec)
  RunTimeRashomon = c(RunTimeRashomon, SimulationResults$run_time)
  rm(SimulationResults)
  }

### All Error Vectors ###
AllErrorVectors = list(ErrorVec_FactorialRandom = ErrorVec_FactorialRandom,
                       ErrorVec_FactorialBreakingTies = ErrorVec_FactorialBreakingTies,
                       ErrorVec_RashomonBreakingTies = ErrorVec_RashomonBreakingTies)
### Get Mean Error ###
MeanErrorVec_FactorialRandom = colMeans(ErrorVec_FactorialRandom)
MeanErrorVec_FactorialBreakingTies = colMeans(ErrorVec_FactorialBreakingTies)
MeanErrorVec_RashomonBreakingTies = colMeans(ErrorVec_RashomonBreakingTies)

list(MeanErrorVec_FactorialRandom = MeanErrorVec_FactorialRandom,
     MeanErrorVec_FactorialBreakingTies = MeanErrorVec_FactorialBreakingTies,
     MeanErrorVec_RashomonBreakingTies = MeanErrorVec_RashomonBreakingTies) -> MeanOutputVector

### Run Time ###
AllRunTimes = list(RunTimeRandom = RunTimeRandom,
                   RunTimeFactorial = RunTimeFactorial,
                   RunTimeRashomon = RunTimeRashomon)

MeanRunTimes = list(MeanRunTimeRandom = mean(RunTimeRandom),
                    MeanRunTimeFactorial = mean(RunTimeFactorial),
                    MeanRunTimeRashomon = mean(RunTimeRashomon))

### Output Vector ###
OutputVector = list(AllErrorVectors = AllErrorVectors,
                    MeanOutputVector = MeanOutputVector,
                    AllRunTimes = AllRunTimes,
                    MeanRunTimes = MeanRunTimes)

input_list = list(MeanErrorVec_FactorialRandom,
                                      MeanErrorVec_FactorialBreakingTies,
                                      MeanErrorVec_RashomonBreakingTies)

save(OutputVector, file= output)



