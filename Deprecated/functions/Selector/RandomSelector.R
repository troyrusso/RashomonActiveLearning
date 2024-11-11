### Summary: Randomly selects data until there is one of each class
### Inputs:
# InitialN: Number of observations to sample at each iteartion
# dat: data set
### Output:
# dat: A training set and a candidate set

RandomSelectorFunc = function(SelectorN, TrainingSet, CandidateSet){
  
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
    
    ### Sample ###
    SelectedIndex = sample(CandidateSet$ID, SelectorN)
    SelectedObservation = CandidateSet[CandidateSet$ID %in% SelectedIndex,]
  
    ### Set Mutation ###
    TrainingSet = rbind(TrainingSet, SelectedObservation)
    CandidateSet = CandidateSet[!(CandidateSet$ID %in% SelectedObservation$ID),]
  
    return(list(TrainingSet = TrainingSet,
                CandidateSet = CandidateSet,
                SelectedObservationID = SelectedObservation$ID))
  }
}
