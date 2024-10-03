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
         DeltaMetric = as.matrix(predict(Model, 
                                                newdata = TrainingSet, 
                                                type = "response"))
         DeltaMetric = cbind(ID = as.numeric(rownames(DeltaMetric)),
                                    Class1 = DeltaMetric[,1],
                                    Class2 = 1-DeltaMetric[,1])
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
         
         DeltaMetric = predict(Model,
                                      newx = as.matrix(TrainingSet[,CovariateList]),
                                      s = MinLambda,
                                      type = "response")
         DeltaMetric = cbind(ID = as.numeric(rownames(DeltaMetric)),
                                            DeltaMetric)
         TrainingPredictedLabels = (ifelse(DeltaMetric > 0.5,1,0))%>% as.factor
         colnames(DeltaMetric) = c("ID", paste0("Class", 1:(ncol(DeltaMetric)-1)))
       },
       Multinomial = {
         # Model and Prediction #
         Model = nnet::multinom(formula = FMLA, 
                                data = TrainingSet,
                                trace = FALSE)
         DeltaMetric = predict(Model,
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
         DeltaMetric = predict(Model,
                                              newx = as.matrix(TrainingSet[, CovariateList]),
                                              s = MinLambda,
                                              type = "response")[,,]
         DeltaMetric =  cbind(ID = as.numeric(rownames(DeltaMetric)),
                                             DeltaMetric)
         colnames(DeltaMetric) = c("ID", paste0("Class", 1:(ncol(DeltaMetric)-1)))
         
         TrainingPredictedLabels = predict(Model,
                                           newx = as.matrix(TrainingSet[, CovariateList]),
                                           s = MinLambda,
                                           type = "class") %>% as.factor
       },
       RandomForest = {
         Model = randomForest::randomForest(formula = FMLA, 
                                            data = TrainingSet)
         DeltaMetric = predict(Model, 
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
         DeltaMetric = ModelPrediction$se.fit
       },
       RashomonLinear = {
         RashomonProfile = RashomonProfileFunc(TrainingSet, CovariateList, LabelName, RashomonParameters)
         TrainingPredictedLabels = RashomonProfile$TrainingPredictedLabels
         DeltaMetric = NULL
         Model = RashomonProfile$RashomonMakeObjects
         RashomonModelLosses = RashomonProfile$RashomonLosses
                },
       Factorial = {
         RashomonProfile = RashomonProfileFunc(TrainingSet, CovariateList, LabelName, RashomonParameters)
         TrainingPredictedLabels = RashomonProfile$TrainingPredictedLabels[,1]
         DeltaMetric = NULL
         Model = RashomonProfile$RashomonMakeObjects
         RashomonModelLosses = RashomonProfile$RashomonLosses[1]
       },
       
  )
  
  
  ReturnList = list(Model = Model,
                    TrainingPredictedLabels = TrainingPredictedLabels,
                    DeltaMetric = DeltaMetric)
  if(ModelType %in% c("RashomonLinear", "Factorial")){
    ReturnList = c(ReturnList, 
                   RashomonModelLosses = list(RashomonModelLosses),
                   RashomonProfile = list(RashomonProfile$RashomonMakeObjects)
                   )}
  
  return(ReturnList)
  }