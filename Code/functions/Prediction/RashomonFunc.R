### Summary: 
### Inputs:
### Output:

RashomonProfileFunc = function(dat, CovariateList, LabelName, RashomonParameters){
  
  ### Parameters ###
  N = nrow(dat)
  M = length(CovariateList)
  K = RashomonParameters$K
  NBins = RashomonParameters$NBins
  
  ### Rashomon Profiles ###
  NewDat = assign_universal_label(dat, arm_cols = CovariateList)
  # NewDat$Y = as.numeric(NewDat$Y)
  StartTime = Sys.time()
  aggregate_rashomon_profiles(data.frame(NewDat),                                  # TrainingSetTrainingData
                              value = LabelName,                       # Response names
                              arm_cols = CovariateList,                # Covariate names
                              M = length(CovariateList),               # Number of covariates
                              H = RashomonParameters$H,                # Maximum number of pools/splits
                              R = RashomonParameters$R,                # Bins of each arm (assume 0 exists)
                              reg = RashomonParameters$reg,            # Penalty on the splits
                              theta = RashomonParameters$theta,        # Threshold
                              inactive = RashomonParameters$inactive
                              ) -> RashomonProfiles  # Losses will always be the last one - (active arms)
  RashomonSetTime = Sys.time() - StartTime
  RashomonSetNumOriginal = length(RashomonProfiles[[1]])
  RashomonMakeObjects = make_rashomon_objects(RashomonProfiles)
  
  ### Rashomon Loss ###
  RashomonLosses = RashomonProfiles[[2]][[length(RashomonProfiles[[2]])]]$losses

  ### Rashomon Prediction ###
  LabeledData = prep_data(data.frame(dat), 
                          CovariateList, 
                          LabelName, 
                          RashomonParameters$R, 
                          drop_unobserved_combinations = TRUE)
  TrainingPredictedLabels = sapply(X = 1:RashomonSetNumOriginal, 
                                   FUN = function(x) predict(RashomonMakeObjects[[x]], LabeledData$universal_label))
    WholeSetTime = Sys.time() - StartTime
    
  ### Limit Rashomon Model Numbers ###
    if(RashomonSetNumOriginal > RashomonParameters$RashomonModelNumLimit & !is.na(RashomonParameters$RashomonModelNumLimit)){
      RashomonLosses = RashomonLosses[1:RashomonParameters$RashomonModelNumLimit]
      RashomonMakeObjects = RashomonMakeObjects[1:RashomonParameters$RashomonModelNumLimit]
      TrainingPredictedLabels = TrainingPredictedLabels[,1:RashomonParameters$RashomonModelNumLimit]
    }
  
  ### Return ###
  return(list(RashomonSetNumOriginal = RashomonSetNumOriginal,
              RashomonSetNumUsed = length(RashomonLosses),
              RashomonLosses = RashomonLosses,
              RashomonMakeObjects = RashomonMakeObjects,
              TrainingPredictedLabels = TrainingPredictedLabels,
              RashomonSetTime = RashomonSetTime,
              WholeSetTime = WholeSetTime))
}
