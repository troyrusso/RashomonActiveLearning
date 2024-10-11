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
  # ValidationFunc(dat, SelectorType, ModelType)
  
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
  # NClass = length(unique(TestSet$Y))
  DeltaMetricVec = numeric(MaxIterationN * nrow(TestSet)) %>% 
    matrix(nrow = MaxIterationN,
           ncol = nrow(TestSet))
  colnames(DeltaMetricVec) = rownames(TestSet)
  ErrorVec = numeric(MaxIterationN)
  # ClassErrorVec = matrix(nrow = MaxIterationN,ncol = NClass)
  # colnames(ClassErrorVec) = paste0("Class", 1:NClass)
  SelectedObservationHistory = numeric(MaxIterationN * SelectorN) %>%
    matrix(nrow = MaxIterationN, 
           ncol = SelectorN)
  if(ModelType == "RashomonLinear"){
    TestSetPrediction = vector(mode = "list", length = MaxIterationN)
    RashomonSetNumList = numeric(length = MaxIterationN*2) %>% 
      matrix(nrow = MaxIterationN,
             ncol =2)
    colnames(RashomonSetNumList) = c("RawNumberofRPSModels", "UsedNumberofRPSModels")
  }else if(ModelType != "RashomonLinear"){TestSetPrediction = DeltaMetricVec}

  ### Progress Bar ###
  pb = txtProgressBar(min = 0, 
                      max = MaxIterationN,
                      style = 3,  
                      width = 50,
                      char = "=")
  start_time = Sys.time()
  
  ### Simulation ###
  for(iter in 1:MaxIterationN){

    ## Progress Bar ##
    setTxtProgressBar(pb, iter)
    
    ## Train Model ##
    ModelTypeSwitchResults = ModelTypeSwitchFunc(TrainingSet, 
                                                 LabelName, 
                                                 CovariateList, 
                                                 ModelType,
                                                 RashomonParameters = RashomonParameters)
    # TrainingSet = data.frame(TrainingSet)[, c("ID", "Y", "YStar", paste0("X",1:length(CovariateList)))]
    TrainingSet = data.frame(TrainingSet)
    Model = ModelTypeSwitchResults$Model
    ModelList[[iter]] = Model
    
    ### Rashomon or not ###
    if(ModelType %in% c("RashomonLinear", "Factorial")){
      RashomonModelLosses = ModelTypeSwitchResults$RashomonModelLosses
      RashomonProfile = ModelTypeSwitchResults$RashomonProfile
      RashomonParameters$RashomonModelLosses = RashomonModelLosses
    }
    
    ### Error and Stopping Criteria ### 
    TestErrorResults = TestErrorFunction(Model, 
                                         ModelType, 
                                         TestSet, 
                                         CovariateList, 
                                         LabelName, 
                                         RashomonParameters)
    if(ModelType == "RashomonLinear"){
      TestSetPrediction[[iter]] = TestErrorResults$TestPredictedLabels
      RashomonSetNumList[iter,] = c(ModelTypeSwitchResults$RashomonSetNumOriginal,
                                    ModelTypeSwitchResults$RashomonSetNumUsed)
    }else if(ModelType != "RashomonLinear"){TestSetPrediction[iter ,] = TestErrorResults$TestPredictedLabels}
    DeltaMetricVec[iter,] = TestErrorResults$DeltaMetric
    ErrorVec[iter] = TestErrorResults$Error
    # ClassErrorVec[iter,] = TestErrorResults$ClassError
    
    ### Selector ###
    SelectorDataSets = SelectorTypeSwitchFunc(ModelType = ModelType,
                                              SelectorType = SelectorType, 
                                              SelectorN = SelectorN,
                                              TestSet = TestSet,
                                              TrainingSet = TrainingSet, 
                                              CandidateSet = CandidateSet,
                                              CovariateList = CovariateList,
                                              DeltaMetric = DeltaMetricVec[iter,])
    ### Set Mutation ###
    TrainingSet = SelectorDataSets$TrainingSet
    CandidateSet = SelectorDataSets$CandidateSet
    SelectedObservationHistory[iter,] = SelectorDataSets$SelectedObservationID
  }
  
  ### System Time ###
  close(pb)
  end_time = Sys.time()
  run_time = end_time - start_time
  
  ### Return ###
  ReturnList = list(ModelList = ModelList,
                    ErrorVec = ErrorVec,
                    DeltaMetricVec = DeltaMetricVec,
                    # ClassErrorVec = ClassErrorVec,
                    SelectorType = SelectorType,
                    ModelType = ModelType,
                    TestSet = TestSet,
                    TestSetPrediction = TestSetPrediction,
                    InitialTrainingSetN = InitialTrainingSetN,
                    InitialCandidateSetN = InitialCandidateSetN,
                    SelectedObservationHistory = SelectedObservationHistory,
                    run_time = run_time)
  
  if(ModelType %in% c("RashomonLinear")){
    ReturnList = c(ReturnList, 
                        RashomonSetNumList = list(RashomonSetNumList)
    )}
  
  return(ReturnList)
}

