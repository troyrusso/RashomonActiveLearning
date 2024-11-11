### Summary:
### Inputs:
### Output:

ActiveLearningPlotFunc = function(SimResults,
                                  StopIter, 
                                  xlower = NULL, 
                                  xupper = NULL){
  ### Set Up ###
  Error = SimResults$Error
  SelectorType = SimResults$SelectorType
  ModelType = SimResults$ModelType
  InitialTrainingSetN = SimResults$InitialTrainingSetN
  if(is.null(xlower)){xlower = 0}
  if(is.null(xupper)){xupper = length(Error)+InitialTrainingSetN}
  
  
  ### Plot ###
  ErrorScatterPlot = ggplot() +
    
    ### Lines ###
    geom_line(mapping = aes(x = (InitialTrainingSetN+1):(length(Error)+InitialTrainingSetN), y = Error)) + 
    geom_vline(xintercept = StopIter, color = "red") + 
    geom_hline(yintercept = Error[StopIter], color = "black", linetype = "dotted", alpha = 0.4) + 
    
    ### Aesthetic ###
    annotate("text", x = 0, y = Error[StopIter], label = round(Error[StopIter],3)) + 
    scale_x_continuous(breaks = c(seq(xlower,xupper, 100), StopIter, InitialTrainingSetN),
                       lim = c(xlower, xupper)) +
    xlab("Number of annotated observations") +
    ylab("Test Set Error") +
    ggtitle(paste0("Selector type and Predictor Type: ", SelectorType, ", ", ModelType))
  
}