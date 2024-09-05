### Summary:
### Inputs:
### Output:

ModelTypeSwitchFunc = function(TrainingSet, 
                               LabelName,
                               CovariateList,
                               ModelType,
                               RashomonParameters = NULL
                               ){

  ### REMEMBER NO INTERCEPT ###
  ### Formula ###
  FMLA = as.formula(paste(LabelName, "~", paste(CovariateList, collapse= "+"), "-1"))
  
  switch(ModelType,
         Logistic = {
         # Model and Prediction #
         Model = glm(FMLA, 
                     data = TrainingSet,
                     family = "binomial")
         TrainingLabelProbabilities = as.matrix(predict(Model, 
                                                newdata = TrainingSet, 
                                                type = "response"))
         TrainingLabelProbabilities = cbind(ID = as.numeric(rownames(TrainingLabelProbabilities)),
                                    Class1 = TrainingLabelProbabilities[,1],
                                    Class2 = 1-TrainingLabelProbabilities[,1])
         TrainingPredictedLabels = 1*(predict(Model, 
                                              newdata = TrainingSet, 
                                              type = "response")>0.5)
       },
       LASSO = {
         # Best Lambda #
         LASSOCV = glmnet(x = as.matrix(TrainingSet[,CovariateList]),
                          y = as.matrix(TrainingSet[,LabelName]),
                          alpha = 1,
                          family = "binomial",
                          intercept = FALSE)
         MinLambda = min(LASSOCV$lambda)
         
         # Model and Prediction #
         Model = glmnet(x = as.matrix(TrainingSet[,CovariateList]),
                        y = as.matrix(TrainingSet[,LabelName]),
                        alpha = 1,
                        lambda = MinLambda,
                        family = "binomial",
                        intercept = FALSE)
         
         TrainingLabelProbabilities = predict(Model,
                                      newx = as.matrix(TrainingSet[,CovariateList]),
                                      s = MinLambda,
                                      type = "response")
         TrainingLabelProbabilities = cbind(ID = as.numeric(rownames(TrainingLabelProbabilities)),
                                            TrainingLabelProbabilities)
         TrainingPredictedLabels = (ifelse(TrainingLabelProbabilities > 0.5,1,0))%>% as.factor
         colnames(TrainingLabelProbabilities) = c("ID", paste0("Class", 1:(ncol(TrainingLabelProbabilities)-1)))
       },
       Multinomial = {
         # Model and Prediction #
         Model = nnet::multinom(formula = FMLA, 
                                data = TrainingSet,
                                trace = FALSE)
         TrainingLabelProbabilities = predict(Model,
                                              newdata = TrainingSet,
                                              type = "prob")
         TrainingPredictedLabels = predict(Model,
                                           newdata = TrainingSet) %>% as.factor
       },
       MultinomLASSO = {
         # Best Lambda #
         LassoRegression = glmnet(x = as.matrix(TrainingSet[, CovariateList]),
                                  y = as.matrix(TrainingSet[,LabelName]),
                                  alpha = 1,
                                  family = "multinomial",
                                  intercept = FALSE)
         MinLambda = min(LassoRegression$lambda)
         
         # Model and Prediction #
         Model = glmnet(x = as.matrix(TrainingSet[, CovariateList]),
                        y = as.matrix(TrainingSet[,LabelName]),
                        alpha = 1,
                        lambda = MinLambda,
                        family = "multinomial",
                        intercept = FALSE)
         TrainingLabelProbabilities = predict(Model,
                                              newx = as.matrix(TrainingSet[, CovariateList]),
                                              s = MinLambda,
                                              type = "response")[,,]
         TrainingLabelProbabilities =  cbind(ID = as.numeric(rownames(TrainingLabelProbabilities)),
                                             TrainingLabelProbabilities)
         colnames(TrainingLabelProbabilities) = c("ID", paste0("Class", 1:(ncol(TrainingLabelProbabilities)-1)))
         
         TrainingPredictedLabels = predict(Model,
                                           newx = as.matrix(TrainingSet[, CovariateList]),
                                           s = MinLambda,
                                           type = "class") %>% as.factor
       },
       RandomForest = {
         Model = randomForest::randomForest(formula = FMLA, 
                                            data = TrainingSet)
         TrainingLabelProbabilities = predict(Model, 
                                              TrainingSet, 
                                              type = "prob")
         TrainingPredictedLabels = predict(Model, 
                                           TrainingSet)
       },
       Linear = {
         # Model and Prediction #
         Model = lm(FMLA, data = TrainingSet)
         ModelPrediction = predict(Model, se.fit =TRUE, data = TraininSet)
         TrainingPredictedLabels = ModelPrediction$fit
         TrainingLabelProbabilities = ModelPrediction$se.fit
       },
       RashomonLinear = {
         RashomonProfile = RashomonProfileFunc(TrainingSet, CovariateList, LabelName, RashomonParameters)
         TrainingPredictedLabels = RashomonProfile$TrainingPredictedLabels
         TrainingLabelProbabilities = NULL
         Model = RashomonProfile$RashomonMakeObjects
         # Model = NULL
         RashomonModelLosses = RashomonProfile$RashomonLosses
         
         # PredictionDifference = abs(TrainingPredictedLabels - data.frame(TrainingSet)[,LabelName])
       },
       Factorial = {
         RashomonProfile = RashomonProfileFunc(TrainingSet, CovariateList, LabelName, RashomonParameters)
         TrainingPredictedLabels = RashomonProfile$TrainingPredictedLabels[,1]
         TrainingLabelProbabilities = NULL
         Model = RashomonProfile$RashomonMakeObjects
         Model = NULL
         RashomonModelLosses = RashomonProfile$RashomonLosses[1]
         
         # PredictionDifference = abs(TrainingPredictedLabels - data.frame(TrainingSet)[,LabelName])
       },
       
  )
  
  
  ReturnList = list(Model = Model,
                    TrainingPredictedLabels = TrainingPredictedLabels,
                    TrainingLabelProbabilities = TrainingLabelProbabilities)
  if(ModelType %in% c("RashomonLinear", "Factorial")){
    ReturnList = c(ReturnList, 
                   RashomonModelLosses = list(RashomonModelLosses),
                   RashomonProfile = list(RashomonProfile$RashomonMakeObjects)
                   )}
  
  return(ReturnList)
  }