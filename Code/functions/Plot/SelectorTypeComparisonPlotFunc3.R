### Summary: 
### Inputs:
### Output:

# NEED TO WORK ON THIS MORE TO GENERALIZE #


SelectorTypeComparisonPlotFunc3 = function(SimulationType1, 
                                          SimulationType2,
                                          SimulationType3,
                                          StopIter1,
                                          StopIter2,
                                          StopIter3,
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
  JointErrors = data.frame(cbind(SimulationType1$Error, SimulationType2$Error, SimulationType3$Error))
  JointErrors$iter = (SimulationType1$InitialTrainingSetN+1):(length(SimulationType1$Error)+SimulationType1$InitialTrainingSetN)
  colnames(JointErrors) = c(paste0(SimulationType1$SelectorType, SimulationType1$ModelType),
                            paste0(SimulationType2$SelectorType, SimulationType2$ModelType),
                            paste0(SimulationType3$SelectorType, SimulationType3$ModelType),
                            "iter")
  JointErrors = pivot_longer(JointErrors, -c(iter))
  colnames(JointErrors) = c("iter", "Method", "value")
  
  JointErrors = JointErrors %>%
    mutate(Method = case_when(Method == "BreakingTiesFactorial" ~ "Naive",
                              Method == "BreakingTiesRashomonLinear" ~ "Rashomon-weighted",
                              Method == "RandomFactorial" ~ "Random"))
  
  # Stop Iter Line
  # JointStopIter = c(SimulationType1$StopIter,SimulationType1$SelectorType,
  #                   SimulationType2$StopIter,SimulationType2$SelectorType) %>%
  #   matrix(nrow = 2, ncol = 2, byrow = TRUE)
  # colnames(JointStopIter) = c("StopIter", "Method")
  # JointStopIter$StopIter = as.numeric(JointStopIter$StopIter)
  
  ### Plot ###
  ErrorScatterPlot = ggplot() +
    
    ## Lines ##
    geom_line(data = JointErrors,
              mapping = aes(x = iter, y = value, color = Method)) + 
    
    ## Aesthetics ##
    scale_x_continuous(breaks = c(seq(xlower,xupper, 50), 
                                  StopIter1,
                                  StopIter2, 
                                  StopIter3, 
                                  SimulationType1$InitialTrainingSetN),
                       lim = c(xlower, xupper))    +
    xlab("Number of annotated observations") +
    ylab("Test Set Error") +
    # ggtitle("Simulation") +
    theme(legend.position = c(0.8,0.8)) +
    theme(plot.title = element_text(size = 15, hjust = 0.5))
  
  return(ErrorScatterPlot)
}
