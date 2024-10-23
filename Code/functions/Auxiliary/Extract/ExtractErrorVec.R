### Set Up ###
rm(list=ls())
directory = "Results/SimulationRaw"
# directory = "/Users/simondn/Documents/RashomonActiveLearning/Results/SimulationRaw/"       ### DELETE ###
library(optparse)   #parse

## Parser ###
option_list = list(
  make_option(c("--JobName"), type = "character", default = NULL, help = "Job Name", metavar = "integer"),
  make_option(c("--RashomonModelNumLimit"), type = "numeric", default = NULL, help = "Max Rashomon number", metavar = "numeric"),
  make_option(c("--Method"), type = "character", default = NULL, help = "Random/Naive/Rashomon", metavar = "numeric"),
  make_option(c("--output"), type = "character", default = NULL, help = "Path to store", metavar = "character")
)
arg.parser = OptionParser(option_list = option_list)
args = parse_args(arg.parser)
RashomonModelNumLimit = args$RashomonModelNumLimit
Method = args$Method
output = args$output

### Auxiliary Function ###
AddRowToMatrix = function(StorageMatrix, NewRow){
  if(ncol(StorageMatrix) == 0){StorageMatrix = matrix(NewRow, nrow = 1, ncol = length(NewRow))
  } else { StorageMatrix = rbind(StorageMatrix, NewRow)}
}

### Data ###
RDataFiles = list.files(path = directory,
                        pattern = paste0(Method, ".*\\.RData$"))
if(Method == "RashomonLinear_BreakingTies"){
  RDataFiles = RDataFiles[grepl(paste0(RashomonModelNumLimit,"_\\.RData$"), 
                                                     RDataFiles)]
}

### Set Up ###
SimulationResultsList = list()
ErrorVec = matrix(ncol= 0, nrow = 0)
RunTime = c()

### Progress Bar ###
pb = txtProgressBar(min = 0, 
                    max = length(RDataFiles),
                    style = 3,  
                    width = 50,
                    char = "=")
### Loop ###
for (i in 1:length(RDataFiles)) {

  ## Progress Bar ##
  setTxtProgressBar(pb, i)
  print(i)
  
  ## Load Data ##
  load(paste0(directory,RDataFiles[i]))
  SimulationResultsList[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec = AddRowToMatrix(ErrorVec,SimulationResults$ErrorVec)
  RunTime = c(RunTime, SimulationResults$run_time)
  rm(SimulationResults)
  }


### Output Vector ###
OutputVector = list(SimulationResultsList = SimulationResultsList,
                    ErrorVec = ErrorVec,
                    RunTime = RunTime)

save(OutputVector, file= output)



