### This will work for now, but should make generalizable with all parameters. ###

### Set Up ###
rm(list=ls())
directory = "Results/Vanilla/"
library(optparse)   #parse

## Parser ###
option_list = list(
  make_option(c("--JobName"), type = "character", default = NULL, help = "Job Name", metavar = "integer"),
  make_option(c("--Method"), type = "character", default = NULL, help = "Random/Naive/Rashomon", metavar = "numeric"),
  make_option(c("--N"), type = "numeric", default = NULL, help = "Number of Observations", metavar = "numeric"),
  make_option(c("--K"), type = "numeric", default = NULL, help = "NumberOfCovariate", metavar = "numeric"),
  make_option(c("--output"), type = "character", default = NULL, help = "Path to store", metavar = "character")
)
arg.parser = OptionParser(option_list = option_list)
args = parse_args(arg.parser)
Method = args$Method
N = args$N
K = args$K
output = args$output

### Auxiliary Function ###
AddRowToMatrix = function(StorageMatrix, NewRow){
  if(ncol(StorageMatrix) == 0){StorageMatrix = matrix(NewRow, nrow = 1, ncol = length(NewRow))
  } else { StorageMatrix = rbind(StorageMatrix, NewRow)}
}

### Data ###
RDataFiles = list.files(path = directory,
                        pattern = ".*\\.RData$")
RDataFiles = RDataFiles[grepl(paste0(Method,"_",N,"_", K, "_"), RDataFiles)]


### Set Up ###
# SimulationResultsList = list()
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
  # SimulationResultsList[[paste0("SimulationResults_", i)]] = SimulationResults
  ErrorVec = AddRowToMatrix(ErrorVec,SimulationResults$ErrorVec)
  RunTime = c(RunTime, SimulationResults$run_time)
  rm(SimulationResults)
  }


### Output Vector ###
OutputVector = list(ErrorVec = ErrorVec,
                    RunTime = RunTime)

save(OutputVector, file= output)



