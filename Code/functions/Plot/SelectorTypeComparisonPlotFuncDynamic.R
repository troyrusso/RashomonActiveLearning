SelectorTypeComparisonPlotFuncDynamic <- function(..., xlower = NULL, xupper = NULL) {
  
  # Capture the list of input vectors and their names
  input_list <- list(...)
  
  # Get the actual names of the input arguments
  var_names <- names(input_list)
  
  # Set x-axis limits
  if(is.null(xlower)) { xlower = 0 }
  if(is.null(xupper)) { xupper = 10 + length(input_list[[1]]) }
  
  # Error Lines: combine all vectors into a data frame
  JointErrors <- data.frame(do.call(cbind, input_list))
  JointErrors$iter <- 1:nrow(JointErrors)
  
  # Pivot to long format for ggplot
  JointErrors <- pivot_longer(JointErrors, -c(iter), names_to = "Method", values_to = "value")
  
  # Replace method names with the input variable names
  JointErrors$Method <- factor(JointErrors$Method, levels = unique(JointErrors$Method), labels = var_names)
  
  # Plot
  ErrorScatterPlot <- ggplot() +
    
    # Lines
    geom_line(data = JointErrors, mapping = aes(x = iter, y = value, color = Method)) + 
    
    # Assuming xlower and xupper are correctly defined as single numeric values
    scale_x_continuous(breaks = c(seq(xlower, xupper, 50), 10, xupper)) +
   
    
    # Aesthetics
    xlab("Number of annotated observations") +
    ylab("Test Set Error") +
    theme(legend.position = c(0.8, 0.8)) +
    theme(plot.title = element_text(size = 15, hjust = 0.5))
  
  return(ErrorScatterPlot)
}
