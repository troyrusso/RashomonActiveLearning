### Summary:
### Inputs:
### Output:

TestErrorFunction = function(Model, 
                             ModelType, 
                             TestSet, 
                             CovariateList, 
                             LabelName, 
                             RashomonParameters){
  
  ### Set Up ###
  NClass = length(unique(TestSet$Y))
  ClassError = numeric(NClass)
  
  ### Switch ###
  switch(ModelType,
         Logistic = {
           DeltaMetric = as.matrix(predict(Model, 
                                                          newdata = TestSet, 
                                                          type = "response"))
           TestPredictedLabels = 1*(predict(Model, 
                                            newdata = TestSet, 
                                            type = "response")>0.5)
           DeltaMetric = cbind(ID = as.numeric(rownames(DeltaMetric)),
                                              Class1 = DeltaMetric[,1], 
                                              Class2 = 1-DeltaMetric[,1])
         },
         LASSO = {
           DeltaMetric = predict(Model,
                                                newx = as.matrix(TestSet[, CovariateList]),
                                                type = "response")
           TestPredictedLabels = ifelse(DeltaMetric > 0.5,1,0)
           
           DeltaMetric = cbind(ID = as.numeric(rownames(DeltaMetric)),
                                              Class1 = DeltaMetric[,1], 
                                              Class2 = 1-DeltaMetric[,1])
         },
         Multinomial = {
           DeltaMetric = predict(Model,
                                                newdata = TestSet,
                                                type = "prob")
           TestPredictedLabels = predict(Model,
                                         newdata = TestSet)
         },
         MultinomLASSO = {
           DeltaMetric = predict(Model,
                                                newx = as.matrix(TestSet[, CovariateList]),
                                                type = "response")[,,]
           TestPredictedLabels = predict(Model,
                                         newx = as.matrix(TestSet[, CovariateList]),
                                         type = "class") %>% as.factor
         },
         RandomForest = {
           TestPredictedLabels = predict(Model, TestSet)
           DeltaMetric = predict(Model, TestSet, type = "prob")
           DeltaMetric = cbind(ID = as.numeric(rownames(DeltaMetric)),
                                              Class1 = DeltaMetric[,1], 
                                              Class2 = 1-DeltaMetric[,1])
         },
         Linear = {
           ModelPrediction = predict(Model, newdata = TestSet, se.fit = TRUE)
           TestPredictedLabels = ModelPrediction$fit
           DeltaMetric = ModelPrediction$se.fit
         },
         RashomonLinear = {
           RashomonSetNum = length(Model)
           NewTestSet = assign_universal_label(TestSet, arm_cols = CovariateList)
           TestSetLabeledData = prep_data(data.frame(TestSet), 
                                          CovariateList, 
                                          LabelName, 
                                          RashomonParameters$R, 
                                          drop_unobserved_combinations = TRUE)
           TestPredictedLabels = sapply(X = 1:RashomonSetNum, 
                                        FUN = function(x) predict(Model[[x]], TestSetLabeledData$universal_label))
           PredictionDifference = (TestPredictedLabels - data.frame(TestSet)[,LabelName])^2
           DifferenceTimesLosses = PredictionDifference %*%  diag(RashomonParameters$RashomonModelLosses)
           DeltaMetric = rowSums(DifferenceTimesLosses)
           # Error = mean((TestPredictedLabels[,1] - data.frame(TestSet)[,LabelName])^2)
           Error = mean((TestPredictedLabels[,1] - data.frame(TestSet)[,LabelName])^2, na.rm = TRUE)
         },
         Factorial={
           NewTestSet = assign_universal_label(TestSet, arm_cols = CovariateList)
           TestSetLabeledData = prep_data(data.frame(TestSet), 
                                          CovariateList, 
                                          LabelName, 
                                          RashomonParameters$R, 
                                          drop_unobserved_combinations = TRUE)
           TestPredictedLabels = predict(Model[[1]], TestSetLabeledData$universal_label)
           
           PredictionDifference = (TestPredictedLabels - data.frame(TestSet)[,LabelName])^2
           DifferenceTimesLosses= PredictionDifference * RashomonParameters$RashomonModelLosses[1]
           DeltaMetric = DifferenceTimesLosses
           Error = mean(PredictionDifference, na.rm = TRUE)
           }
  )
  
  
  return(list(Error = Error,
              TestPredictedLabels = TestPredictedLabels,
              DeltaMetric = DeltaMetric))
}
