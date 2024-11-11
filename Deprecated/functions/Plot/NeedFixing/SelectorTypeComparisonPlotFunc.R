### Summary: 
### Inputs:
### Output:

# NEED TO WORK ON THIS MORE TO GENERALIZE #


SelectorTypeComparisonPlotFunc = function(SimulationType1, 
                                          SimulationType2,
                                          StopIter1,
                                          StopIter2,
                                          xlower = NULL, 
                                          xupper = NULL){
  ### Validation ###
  if(SimulationType1$InitialTrainingSetN != SimulationType2$InitialTrainingSetN){
    stop("Starting points are not the same.")
  }
  
  if(length(SimulationType1$Error) != length(SimulationType2$Error)){
    stop("Ending points are not the same.")
  }  
  
  if((SimulationType1$Error[length(SimulationType1$Error)] - SimulationType2$Error[length(SimulationType2$Error)])>1e-5){
    warning("The error rate with all observations labelled are not the same.")
  }
  
  ### Set Up ###
  if(is.null(xlower)){xlower = 0}
  if(is.null(xupper)){xupper = length(SimulationType1$Error)+SimulationType1$InitialTrainingSetN}
  
  
  # Error Lines #
  JointErrors = data.frame(cbind(SimulationType1$Error, SimulationType2$Error))
  JointErrors$iter = (SimulationType1$InitialTrainingSetN+1):(length(SimulationType1$Error)+SimulationType1$InitialTrainingSetN)
  colnames(JointErrors) = c(SimulationType1$SelectorType, SimulationType2$SelectorType, "iter")
  JointErrors = pivot_longer(JointErrors, c(Random, BreakingTies))
  colnames(JointErrors) = c("iter", "SelectorType", "value")
  
  # Stop Iter Line
  # JointStopIter = c(SimulationType1$StopIter,SimulationType1$SelectorType,
  #                   SimulationType2$StopIter,SimulationType2$SelectorType) %>%
  #   matrix(nrow = 2, ncol = 2, byrow = TRUE)
  # colnames(JointStopIter) = c("StopIter", "SelectorType")
  # JointStopIter$StopIter = as.numeric(JointStopIter$StopIter)
  
  ### Plot ###
  ErrorScatterPlot = ggplot() +
    
    ## Lines ##
    geom_line(data = JointErrors,
              mapping = aes(x = iter, y = value, linetype = SelectorType)) + 
    
    ## Iteration Stop Line ##
    geom_vline(xintercept = StopIter1,
               color = "red",
               linetype = "dashed") + 
    
    geom_vline(xintercept = StopIter2,
               color = "red",
               linetype = "solid") + 
    
    ## Aesthetics ##
    scale_x_continuous(breaks = c(seq(xlower,xupper, 100), 
                                  StopIter1,
                                  StopIter2, 
                                  SimulationType1$InitialTrainingSetN),
                       lim = c(xlower, xupper))    +
    xlab("Number of annotated observations") +
    ylab("Test Set Error") +
    ggtitle("Simulation by Error") +
    theme(plot.title = element_text(size = 15, hjust = 0.5))
  
  return(ErrorScatterPlot)
}