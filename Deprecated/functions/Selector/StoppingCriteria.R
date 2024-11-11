### Summary:
### Inputs:
### Output:

# StoppingCriteriaFunc = function(ErrorVector, ErrorThreshold, VarThreshold, TailN){
#   
#   if(length(ErrorVector) >= TailN){
#     
#     ### Condition 1: Low Error ###
#     if(all(tail(ErrorVector, TailN) <= ErrorThreshold)){Cond1 = TRUE}else{Cond1 = FALSE}
#     
#     ### Condition 2: Low Variation ###
#     if(var(tail(ErrorVector, TailN)) <= VarThreshold){Cond2 = TRUE}else{Cond2 = FALSE}
#   }else{
#     Cond1 = FALSE
#     Cond2 = FALSE
#   }
#   
#   ### Return ###
#   if(Cond1 & Cond2){return(IterStop = length(ErrorVector))}else{return(NULL)}
# }

StoppingCriteriaFunc = function(SimulationType, ErrorThreshold, VarThreshold, TailN){

RollingVariance = sapply(X = TailN:length(SimulationType$Error),
                         FUN = function(i) var(SimulationType$Error[(i-TailN+1):i]))

RollingError = sapply(X = TailN:length(SimulationType$Error),
                      FUN = function(i) max(SimulationType$Error[(i-TailN+1):i]))

RollVarIndx = which(RollingVariance <= VarThreshold)
RollErrorIndx = which(RollingError <= ErrorThreshold)

StopIter = intersect(RollVarIndx,RollErrorIndx)[1]+TailN +SimulationType$InitialTrainingSetN

return(StopIter)

}

