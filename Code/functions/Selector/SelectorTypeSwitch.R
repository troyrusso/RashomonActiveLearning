### Summary:
### Inputs:
### Output:

SelectorTypeSwitchFunc = function(ModelType,
                                  SelectorType, 
                                  SelectorN, 
                                  TestSet,
                                  TrainingSet, 
                                  CandidateSet, 
                                  CovariateList,
                                  DeltaMetric){
  switch(SelectorType,
         Random = {
           SelectorDataSets = RandomSelectorFunc(SelectorN = SelectorN,
                                                 TrainingSet = TrainingSet, 
                                                 CandidateSet = CandidateSet)},
         BreakingTies = {
           SelectorDataSets = BreakingTiesSelectorFunc(ModelType = ModelType,
                                                       DeltaMetric = DeltaMetric,
                                                       TestSet = TestSet,
                                                       TrainingSet = TrainingSet,
                                                       CandidateSet = CandidateSet,
                                                       CovariateList,
                                                       SelectorN = SelectorN)})
  
  return(SelectorDataSets)
}