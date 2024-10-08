

## Load Functions

rm(list=ls())
directory = "/Users/simondn/Documents/RashomonActiveLearning/"

if(exists("directory")){
  source(paste0(directory,"Code/functions/Auxiliary/LoadFunctions.R"))
}else if(!exists("directory")){source("Code/functions/Main/LoadFunctions.R")}

# Inputs

### Input ###
N = 300
K = 4
NClass = 2
# ClassProportion = c(3/5, 2/5)
ClassProportion = rep(1/NClass, NClass)
CovCorrVal = 0
TestProportion = 0.2
SelectorN = 1
InitialN = 10
NBins = 3


### Data ###
DGPResults = GenerateDataFunc(N, K, NClass, ClassProportion, CovCorrVal, NBins = NBins)
dat = DGPResults$dat
TrueBetas = DGPResults$TrueBetas
# dat$Y = as.numeric(dat$Y)
CovariateList = c("X1", "X2", "X3", "X4")
RashomonParameters = list(K = K, 
                          NBins = NBins,
                          H = Inf,                           # Maximum number of pools/splits
                          R = NBins+1,                       # Bins of each arm (assume 0 exists)
                          reg = 0.1,                         # Penalty on the splits
                          theta = 3,                         # Threshold; determine relative to best model
                          inactive = 0)


## Algorithm
LabelName = "YStar"
SelectorN = 1


for(seed in 1:10){
  print(paste0("Seed: ", seed))
  ### Seed ###
  set.seed(seed)
  
  ### Generate Data ###
  DGPResults = GenerateDataFunc(N, K, NClass, ClassProportion, CovCorrVal, NBins = NBins)
  dat = DGPResults$dat
  TrueBetas = DGPResults$TrueBetas
  
  ### Random ###
  tryCatch({
  SimulationFunc(dat = dat,
                 LabelName = LabelName,
                 CovariateList = CovariateList,
                 TestProportion = TestProportion,
                 SelectorType = "Random",
                 SelectorN = SelectorN,
                 ModelType = "Factorial",
                 InitialN = InitialN,
                 RashomonParameters = RashomonParameters,
                 seed = seed) -> SimResultsBreakingTiesRandom
  }, error = function(e) {
    message(paste("Error in Random for seed", seed, ":", e))
    next
  })
  
  ### Naive ###
    tryCatch({
  SimulationFunc(dat = dat,
                 LabelName = LabelName,
                 CovariateList = CovariateList,
                 TestProportion = TestProportion,
                 SelectorType = "BreakingTies",
                 SelectorN = SelectorN,
                 ModelType = "Factorial",
                 InitialN = InitialN,
                 RashomonParameters = RashomonParameters,
                 seed = seed) -> SimResultsBreakingTiesFactorial
    }, error = function(e) {
      message(paste("Error in Random for seed", seed, ":", e))
      next
    })

  ### Rashomon ###
      tryCatch({
  SimulationFunc(dat = dat,
                 LabelName = LabelName,
                 CovariateList = CovariateList,
                 TestProportion = TestProportion,
                 SelectorType = "BreakingTies",
                 SelectorN = SelectorN,
                 ModelType = "RashomonLinear",
                 InitialN = InitialN,
                 RashomonParameters = RashomonParameters,
                 seed = seed) -> SimResultsBreakingTiesRashomon
      }, error = function(e) {
        message(paste("Error in Random for seed", seed, ":", e))
        next
      })
  
  ### Save ###
  SimResultsBreakingTiesFactorial$Parameters = list(seed = seed,
                                                    N = N,
                                                    K = K,
                                                    NClass = NClass,
                                                    ClassProportion = ClassProportion,
                                                    CovCorrVal = CovCorrVal,
                                                    TestProportion = TestProportion,
                                                    SelectorN = SelectorN,
                                                    InitialN = InitialN,
                                                    NBins = NBins,
                                                    RashomonParameters = RashomonParameters)
  SimResultsBreakingTiesRashomon$Parameters = list(seed = seed,
                                                   N = N,
                                                   K = K,
                                                   NClass = NClass,
                                                   ClassProportion = ClassProportion,
                                                   CovCorrVal = CovCorrVal,
                                                   TestProportion = TestProportion,
                                                   SelectorN = SelectorN,
                                                   InitialN = InitialN,
                                                   NBins = NBins,
                                                   RashomonParameters = RashomonParameters)
  SimResultsBreakingTiesRandom$Parameters = list(seed = seed,
                                                 N = N,
                                                 K = K,
                                                 NClass = NClass,
                                                 ClassProportion = ClassProportion,
                                                 CovCorrVal = CovCorrVal,
                                                 TestProportion = TestProportion,
                                                 SelectorN = SelectorN,
                                                 InitialN = InitialN,
                                                 NBins = NBins,
                                                 RashomonParameters = RashomonParameters)

  ### Save Results and Parameters ###
    save(SimResultsBreakingTiesFactorial,
       file = paste0(directory,"Results/NaiveResults_Seed",seed,".RData"))
  save(SimResultsBreakingTiesRashomon,
       file = paste0(directory,"Results/RashomonResults_Seed",seed,".RData"))
  save(SimResultsBreakingTiesRandom,
       file = paste0(directory,"Results/RandomResults_Seed",seed,".RData"))

}

SimResultsBreakingTiesFactorialList = vector(mode = "list", length = 7)
SimResultsBreakingTiesRashomonList = vector(mode = "list", length = 7)
SimResultsBreakingTiesRandomList = vector(mode = "list", length = 7)

for(i in 1:7){
  print(paste0("Iteration: ", i))
  
  load(paste0("/Users/simondn/Documents/RashomonActiveLearning/Results/NaiveResults_Seed", i, ".RData"))
  load(paste0("/Users/simondn/Documents/RashomonActiveLearning/Results/RashomonResults_Seed", i, ".RData"))
  load(paste0("/Users/simondn/Documents/RashomonActiveLearning/Results/RandomResults_Seed", i, ".RData"))
  
  SimResultsBreakingTiesFactorialList[[i]] = SimResultsBreakingTiesFactorial
  SimResultsBreakingTiesRashomonList[[i]] = SimResultsBreakingTiesRashomon
  SimResultsBreakingTiesRandomList[[i]] = SimResultsBreakingTiesRandom
  

  rm(SimResultsBreakingTiesFactorial)
  rm(SimResultsBreakingTiesRashomon)
  rm(SimResultsBreakingTiesRandom)
}

rbind(SimResultsBreakingTiesFactorialList[[1]]$ErrorVec,
      SimResultsBreakingTiesFactorialList[[2]]$ErrorVec,
      SimResultsBreakingTiesFactorialList[[3]]$ErrorVec,
      SimResultsBreakingTiesFactorialList[[4]]$ErrorVec,
      SimResultsBreakingTiesFactorialList[[5]]$ErrorVec,
      SimResultsBreakingTiesFactorialList[[6]]$ErrorVec,
      SimResultsBreakingTiesFactorialList[[7]]$ErrorVec) -> FactorialAllResults

write.table(FactorialAllResults, file = '/Users/simondn/Documents/RashomonActiveLearning/Results/FactorialAllResults.csv', 
            sep = ',', 
            row.names=FALSE, 
            col.names=FALSE)

rbind(SimResultsBreakingTiesRashomonList[[1]]$ErrorVec,
      SimResultsBreakingTiesRashomonList[[2]]$ErrorVec,
      SimResultsBreakingTiesRashomonList[[3]]$ErrorVec,
      SimResultsBreakingTiesRashomonList[[4]]$ErrorVec,
      SimResultsBreakingTiesRashomonList[[5]]$ErrorVec,
      SimResultsBreakingTiesRashomonList[[6]]$ErrorVec,
      SimResultsBreakingTiesRashomonList[[7]]$ErrorVec) -> RashomonAllResults

write.table(RashomonAllResults, file = '/Users/simondn/Documents/RashomonActiveLearning/Results/RashomonAllResults.csv', 
            sep = ',', 
            row.names=FALSE, 
            col.names=FALSE)

rbind(SimResultsBreakingTiesRandomList[[1]]$ErrorVec,
      SimResultsBreakingTiesRandomList[[2]]$ErrorVec,
      SimResultsBreakingTiesRandomList[[3]]$ErrorVec,
      SimResultsBreakingTiesRandomList[[4]]$ErrorVec,
      SimResultsBreakingTiesRandomList[[5]]$ErrorVec,
      SimResultsBreakingTiesRandomList[[6]]$ErrorVec,
      SimResultsBreakingTiesRandomList[[7]]$ErrorVec) -> RandomAllResults

write.table(RandomAllResults, file = '/Users/simondn/Documents/RashomonActiveLearning/Results/RandomAllResults.csv', 
            sep = ',', 
            row.names=FALSE, 
            col.names=FALSE)

       