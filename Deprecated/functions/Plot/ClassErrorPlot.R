### Summary: 
### Inputs:
### Output:

# NEED TO WORK ON THIS MORE TO GENERALIZE #

ClassErrorPlotFunc = function(SimulationResults, xlower = NULL, xupper = NULL){
  
  ### Set Up ###
  if(is.null(xlower)){xlower = 0}
  if(is.null(xupper)){xupper = length(SimulationResults$Error) + SimulationResults$InitialTrainingSetN}
  ClassErrorDat = data.frame(SimulationResults$ClassError) %>% 
    mutate(iter = (SimulationResults$InitialTrainingSetN+1):(nrow(SimulationResults$ClassError)+SimulationResults$InitialTrainingSetN)) %>%
    pivot_longer(-iter)
  colnames(ClassErrorDat) = c("iter", "Class", "error")

  ### Plot ###
  ErrorClassPlot = ggplot() + 
    geom_line(data = ClassErrorDat,
              mapping = aes(x = iter, y = error, color = Class)) +
    scale_x_continuous(breaks = c(seq(xlower,xupper, 100), SimulationResults$InitialTrainingSetN),
                       lim = c(xlower, xupper)) +
    xlab("Number of annotated observations") +
    ylab("Test Set Error") + 
    ggtitle(paste0("Selector type: ", SimulationResults$SelectorType))
  
  
  return(ErrorClassPlot)
}