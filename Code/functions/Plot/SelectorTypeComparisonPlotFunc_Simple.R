### Summary: 
### Inputs:
### Output:

# NEED TO WORK ON THIS MORE TO GENERALIZE #


SelectorTypeComparisonPlotFuncSimple = function(SimulationTypeErrorVec1, 
                                                 SimulationTypeErrorVec2,
                                                 SimulationTypeErrorVec3,
                                                 xlower = NULL, 
                                                 xupper = NULL){
  ### Set Up ###
  if(is.null(xlower)){xlower = 0}
  if(is.null(xupper)){xupper = 10+ncol(SimulationTypeErrorVec1)}
  
  # Error Lines #
  JointErrors = data.frame(cbind(SimulationTypeErrorVec1, SimulationTypeErrorVec2))
  JointErrors$iter = 1:length(SimulationTypeErrorVec1)
  # colnames(JointErrors) = c("Naive", "Rashomon-weighted", "Random", "iter")
  JointErrors = pivot_longer(JointErrors, -c(iter))
  colnames(JointErrors) = c("iter", "Method", "value")
  
  ### Plot ###
  ErrorScatterPlot = ggplot() +
    
    ## Lines ##
    geom_line(data = JointErrors,
              mapping = aes(x = iter, y = value, color = Method)) + 
    
    ## Aesthetics ##
    # scale_x_continuous(breaks = c(seq(xlower,xupper, 1)),
    #                    lim = c(xlower, xupper))    +
    xlab("Number of annotated observations") +
    ylab("Test Set Error") +
    # ggtitle("Simulation") +
    theme(legend.position = c(0.8,0.8)) +
    theme(plot.title = element_text(size = 15, hjust = 0.5))
  
  return(ErrorScatterPlot)
}
