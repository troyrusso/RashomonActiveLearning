### Summary:
### Inputs:
### Output:

BreakingTiesSelectorFunc = function(DeltaMetric, 
                                    TestSet, 
                                    TrainingSet, 
                                    CandidateSet, 
                                    CovariateList,
                                    ModelType,
                                    SelectorN){
  
  ### Last Observations ###
  if(nrow(CandidateSet) <= SelectorN){
    
    SelectedObservation = CandidateSet
    SelectedObservationID = c(SelectedObservation$ID, rep(NA,SelectorN - nrow(CandidateSet)))
    TrainingSet = rbind(TrainingSet, SelectedObservation)
    CandidateSet = CandidateSet[CandidateSet$ID != SelectedObservation$ID,]
    
    return(list(TrainingSet = TrainingSet,
                CandidateSet = CandidateSet,
                SelectedObservationID = SelectedObservationID))
  }else{
  
  ### Most Uncertain Observation ###
    MostUncertainObsResults = MostUncertainObservationsFunc(DeltaMetric = DeltaMetric, 
                                                     TestSet = TestSet, 
                                                     CandidateSet = CandidateSet,
                                                     CovariateList,
                                                     ModelType = ModelType)
    MostUncertainObsID = MostUncertainObsResults$MostUncertainObsID
    MostUncertainObs = MostUncertainObsResults$MostUncertainObsCovariates
    
  ### Distance Metric###
    
    ## Norm ##
  (dist(rbind(MostUncertainObs[, CovariateList],CandidateSet[, CovariateList]),
        method = "euclidean") %>%
    as.matrix)[,1] -> NormDistance
    NormDistance = NormDistance[names(NormDistance) != rownames(MostUncertainObs)]

  #   ## Mahlanobis Distance ##
  # VarCovMatrix = cov(rbind(MostUncertainObs,
  #                          CandidateSet[, CovariateList]))
  # MalDistance = try(stats::mahalanobis(x = as.matrix(CandidateSet[, CovariateList]),
  #                                      center = as.matrix(MostUncertainObs),  
  #                                      cov = VarCovMatrix),
  #                   silent = TRUE)
  #   
  #   if(inherits(MalDistance, 'try-error')){
  #     print("MalDistance Error")
  #     return(list(TrainingSet = TrainingSet,
  #                 CandidateSet = CandidateSet))
  #   }else{
  #     CandidateSet$MalDistance = MalDistance
  #   }
  
  ## Set Distance ##
  DistanceMetric = NormDistance
  CandidateSet$DistanceMetric = DistanceMetric
  CandidateSet$YDifference = abs(CandidateSet$Y - filter(TestSet, ID == MostUncertainObsID)$Y)

  ### Matched Candidate Data ###
  ### If there are more than 1 candidate observations that have the same factorial design, 
  ### choose the candidate observation with the most different response Y than the
  ### most uncertain test observation.
  SelectedObservation = CandidateSet[order(CandidateSet[,"DistanceMetric"], 
                                              -CandidateSet[,"YDifference"]), ][1:SelectorN,]
  SelectedObservationID = SelectedObservation$ID

  ### Set Mutation ###
  TrainingSet = rbind(data.frame(TrainingSet)[, setdiff(names(TrainingSet), c("BreakingTiesProb"))], 
                      SelectedObservation[, setdiff(names(SelectedObservation), c("DistanceMetric", "YDifference"))])
  
  CandidateSet = CandidateSet[!(CandidateSet$ID %in% SelectedObservation$ID), 
                              setdiff(names(CandidateSet), c("DistanceMetric",
                                                             "YDifference"))]
  
  return(list(TrainingSet = TrainingSet,
              CandidateSet = CandidateSet,
              SelectedObservationID = SelectedObservation$ID))
  }
}



