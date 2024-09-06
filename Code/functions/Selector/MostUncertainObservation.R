### Summary:
### Inputs:
### Output:

MostUncertainObservationsFunc = function(DeltaMetric, 
                                         TestSet, 
                                         CandidateSet, 
                                         ModelType,
                                         CovariateList){
  
  if(ModelType %in% c("Linear","RashomonLinear", "Factorial")){
    TestSet = cbind(TestSet, DeltaMetric)
    IDRec = arrange(TestSet, desc(DeltaMetric))$ID[1]
    MostUncertainObs = data.frame(TestSet)[TestSet$ID %in% IDRec, CovariateList]
  }else if(!(ModelType %in% c("Linear","RashomonLinear", "Factorial"))){
    ProbMax1 = apply(X = DeltaMetric[, setdiff(colnames(DeltaMetric), "ID")], 
                     MARGIN = 1, 
                     FUN = max)
    ProbMax2 = apply(X = DeltaMetric[, setdiff(colnames(DeltaMetric), "ID")], 
                     MARGIN = 1, 
                     FUN = function(x) {sort(x,partial=length(x)-1)[length(x)-1]})
    TestSet$BreakingTiesProb = ProbMax1 - ProbMax2
    IDRec = arrange(TestSet, BreakingTiesProb)$ID[1]
    MostUncertainObs = TestSet[TestSet$ID %in% IDRec, CovariateList]
  }
  return(MostUncertainObs)
  
}
  
  
  
  
  
  