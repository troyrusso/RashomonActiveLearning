SimulationFunc = function(dat,
                          LabelName,
                          CovariateList,
                          TestProportion = 0.2,
                          SelectorType,
                          SelectorN,
                          ModelType,
                          InitialN,
                          RashomonParameters = NULL,
                          seed){
  ### Seed ###
  set.seed(seed)
  
  ### Validation ###
  ValidationFunc(dat, SelectorType, ModelType)
  
  ### Train Test Split ###
  TestSize = floor(TestProportion * nrow(dat))
  TrainingIndices = sample(seq_len(nrow(dat)), size = nrow(dat) - TestSize)
  TestSet = dat[-TrainingIndices,]
  dat = dat[TrainingIndices,]
  # dat$Y = as.numeric(dat$Y)

  ### Random Start ###
  RandomStart = RandomStartFunc(InitialN=InitialN, dat=dat)
  TrainingSet = RandomStart$TrainingSet
  CandidateSet = RandomStart$CandidateSet
  InitialTrainingSetN = nrow(TrainingSet)
  InitialCandidateSetN = nrow(CandidateSet)
  
  ### Set Up ###
  MaxIterationN = ceiling(nrow(CandidateSet)/SelectorN)
  ModelList = vector('list', MaxIterationN)
  NClass = length(unique(TestSet$Y))
  TestSetPrediction = numeric(MaxIterationN * nrow(TestSet)) %>% 
    matrix(nrow = MaxIterationN,
           ncol = nrow(TestSet))
  colnames(TestSetPrediction) = rownames(TestSet)
  ErrorVec = numeric(MaxIterationN)
  ClassErrorVec = matrix(nrow = MaxIterationN,
                      ncol = NClass)
  colnames(ClassErrorVec) = paste0("Class", 1:NClass)
  SelectedObservationHistory = numeric(MaxIterationN * SelectorN) %>%
    matrix(nrow = MaxIterationN,
           ncol = SelectorN)
  DeltaMetricVec = ErrorVec

  ### Progress Bar ###
  pb = txtProgressBar(min = 0, 
                      max = MaxIterationN,
                      style = 3,  
                      width = 50,
                      char = "=")
  start_time = Sys.time()
  
  ### Simulation ###
  for(iter in 1:MaxIterationN){

    # print(iter)
    
    ## Progress Bar ##
    setTxtProgressBar(pb, iter)
    
    ## Train Model ##
    ModelTypeSwitchResults = ModelTypeSwitchFunc(TrainingSet, 
                                                 LabelName, 
                                                 CovariateList, 
                                                 ModelType,
                                                 RashomonParameters = RashomonParameters)
    TrainingSet = data.frame(TrainingSet)[, c("ID", "Y", "YStar", paste0("X",1:length(CovariateList)))]
    Model = ModelTypeSwitchResults$Model
    ModelList[[iter]] = Model
    
    ### Rashomon or not ###
    if(ModelType %in% c("RashomonLinear", "Factorial")){

    ### Error and Stopping Criteria ### 
    
    ### START FOR NOW - LOOK THIS SHIT OVER HAHAHAHA ;-; .-. D: ###
      RashomonModelLosses = ModelTypeSwitchResults$RashomonModelLosses
      RashomonProfile = ModelTypeSwitchResults$RashomonProfile
      TestSet = TrainingSet                                                                
      TestPredictedLabels = ModelTypeSwitchResults$TrainingPredictedLabels
      # PredictionDifference = (TestPredictedLabels - data.frame(TestSet)[,LabelName])^2
      
      if(length(RashomonModelLosses) ==1){
        PredictionDifference = (TestPredictedLabels - data.frame(TestSet)[,LabelName])^2
        DifferenceTimesLosses= PredictionDifference * RashomonModelLosses
        DeltaMetric = DifferenceTimesLosses
        ErrorVec[iter] = mean(PredictionDifference)
        DeltaMetricVec[iter] = max(DeltaMetric)
        }else if(length(RashomonModelLosses) > 1){
          PredictionDifference = (TestPredictedLabels - data.frame(TestSet)[,LabelName])^2
          DifferenceTimesLosses= PredictionDifference %*%  diag(RashomonModelLosses)
          DeltaMetric = max(rowSums(DifferenceTimesLosses))
          ErrorVec[iter] = mean((TestPredictedLabels[,1] - data.frame(TestSet)[,LabelName])^2)
          DeltaMetricVec[iter] = max(DeltaMetric)
        }
    
      ClassErrorVec[iter,] = tapply(X = 1:length(TestSet$Y), 
                          INDEX = TestSet$Y, 
                          FUN = function(i) mean((TestPredictedLabels[i] - TestSet$YStar[i])^2)) %>%
        as.vector
      
    }else if(!ModelType %in% c("RashomonLinear", "Factorial")){                                               # DELETE LATER
      TestErrorResults = TestErrorFunction(Model, ModelType, TestSet, CovariateList, LabelName)
      TestSetPrediction[iter ,] = TestErrorResults$TestPredictedLabels
      DeltaMetric = TestErrorResults$TestPredictedProbabilities
      ErrorVec[iter] = TestErrorResults$Error
      ClassError[iter,] = TestErrorResults$ClassError
      }
    ### END FOR NOW ###    TestSetPrediction[iter ,] = TestErrorResults$TestPredictedLabels
    
    # TestErrorResults = TestErrorFunction(Model, ModelType, TestSet, CovariateList, LabelName)
    # TestSetPrediction[iter ,] = TestErrorResults$TestPredictedLabels
    # DeltaMetric = TestErrorResults$TestPredictedProbabilities
    # ErrorVec[iter] = TestErrorResults$Error
    # ClassErrorVec[iter,] = TestErrorResults$ClassError
    
    # print(paste0("[Iter ", iter, "] Loss: ", ErrorVec[iter]))
    

    ### Selector ###
    SelectorDataSets = SelectorTypeSwitchFunc(ModelType = ModelType,
                                              SelectorType = SelectorType, 
                                              SelectorN = SelectorN,
                                              TestSet = TestSet,
                                              TrainingSet = TrainingSet, 
                                              CandidateSet = CandidateSet,
                                              CovariateList = CovariateList,
                                              DeltaMetric = DeltaMetric)
    ### Set Mutation ###
    TrainingSet = SelectorDataSets$TrainingSet
    CandidateSet = SelectorDataSets$CandidateSet
    SelectedObservationHistory[iter,] = SelectorDataSets$SelectedObservationID
      }
  
  ### System Time ###
  close(pb)
  end_time = Sys.time()
  run_time = end_time - start_time

  return(list(ModelList = ModelList,
              ErrorVec = ErrorVec,
              DeltaMetricVec = DeltaMetricVec,
              ClassErrorVec = ClassErrorVec,
              SelectorType = SelectorType,
              ModelType = ModelType,
              TestSet = TestSet,
              TestSetPrediction = TestSetPrediction,
              InitialTrainingSetN = InitialTrainingSetN,
              InitialCandidateSetN = InitialCandidateSetN,
              SelectedObservationHistory = SelectedObservationHistory,
              run_time = run_time))
}

