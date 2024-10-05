### Summary: Randomly selects data until there is one of each class
### Inputs:
# InitialN: Number of observations in each class
# dat: data set
### Output:
# dat: A training set and a candidate set

RandomStartFunc = function(InitialN = 1, dat){
  FMLA = paste("Y ~ ", paste0())

  ### Set Up ###
  N = nrow(dat)
  Classes = sort(unique(dat$Y))

  TrainingIndices = sample(1:N, InitialN)
  CandidateIndices = setdiff(1:N, TrainingIndices)
  TrainingSet = dat[TrainingIndices, ]
  CandidateSet = dat[CandidateIndices, ]
  
  # ### CountPerClass
  # CountPerClass <- function(dat, Classes) {
  #   sapply(Classes, function(cls) sum(dat$Y == cls))
  # }
  # 
  # 
  # ### While Loop ###
  # while(any(CountPerClass(TrainingSet, Classes) < InitialN)){
  #   
  #   ## Draw ##
  #   TrainingIndices = c(TrainingIndices, sample(CandidateIndices, SampleN))
  #   CandidateIndices = setdiff(CandidateIndices, TrainingIndices)
  #   TrainingSet = dat[TrainingIndices, ]
  #   CandidateSet = dat[CandidateIndices, ]
  #   
  # }

  return(list(TrainingSet = TrainingSet,
              CandidateSet = CandidateSet))
}
