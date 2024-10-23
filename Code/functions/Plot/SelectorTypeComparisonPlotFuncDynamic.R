SelectorTypeComparisonPlotFuncConfidenceBands <- function(...) {
  
  # Capture the list of input vectors and their names
  input_list <- list(...)
  
  # Get the actual names of the input arguments
  var_names <- names(input_list)
  
  # Set x-axis limits
  xlower = 0
  xupper = 10 + ncol(input_list[[1]]) # Adjust for your dataset's dimensions
  
  # Error Lines
  JointErrors <- lapply(input_list, function(x) {
    data.frame(mean = apply(x, 2, mean), 
               lower = apply(x, 2, function(y) mean(y) - qt(0.975, df=length(y)-1) * sd(y) / sqrt(length(y))),
               upper = apply(x, 2, function(y) mean(y) + qt(0.975, df=length(y)-1) * sd(y) / sqrt(length(y))),
               iter = 1:ncol(x))  # Assuming each column is a separate time point
  })
  
  # Combine into a single data frame with method names
  JointErrors <- do.call(rbind, Map(function(df, method) {
    df$Method <- method
    df
  }, JointErrors, var_names))
  
  # Plot
  ErrorScatterPlot <- ggplot() +
    
    # Lines for means
    geom_line(data = JointErrors, mapping = aes(x = iter, y = mean, color = Method)) + 
    
    # Confidence intervals
    geom_ribbon(data = JointErrors, mapping = aes(x = iter, ymin = lower, ymax = upper, fill = Method), alpha = 0.2) +
    
    # Assuming xlower and xupper are correctly defined as single numeric values
    scale_x_continuous(breaks = c(seq(xlower, xupper, 50), 10, xupper)) +
    
    # Aesthetics
    xlab("Number of annotated observations") +
    ylab("Test Set Error") +
    theme(legend.position = c(0.8, 0.8)) +
    theme(plot.title = element_text(size = 15, hjust = 0.5))
  
  return(ErrorScatterPlot)
}

SelectorTypeComparisonPlotFunc <- function(...) {
  
  # Capture the list of input vectors and their names
  input_list <- list(...)
  
  # Get the actual names of the input arguments
  var_names <- names(input_list)
  
  # Set x-axis limits
  xlower = 0
  xupper = 10 + length(input_list[[1]])
  
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