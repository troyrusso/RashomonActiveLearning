### Summary:
### Inputs:
### Output:


ValidationFunc = function(dat, SelectorType, ModelType){
  
  SelectorTypeList = c("Random", "BreakingTies")
  ModelTypeList = c("Logistic", "LASSO", "Multinomial", "MultinomLASSO", "RandomForest", "Linear", "RashomonLinear", "Factorial")
  
  if(!(SelectorType %in% SelectorTypeList)){
  stop("SelectorType not supported.")}
  
  if(!(ModelType %in% ModelTypeList)){
  stop("ModelType not supported.")
  }
  
  if(length(unique(dat$Y)) > 2 & ModelType %in% c("Logistic", "LASSO")){
    stop(paste0("There are ", length(unique(dat$Y)), " classes. Change ModelType to Multinomial "))
  }
  
}