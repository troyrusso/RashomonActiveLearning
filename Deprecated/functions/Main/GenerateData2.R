# Load necessary library for visualization
if (!requireNamespace("plotly", quietly = TRUE)) install.packages("plotly")
library(plotly)

# Define more complex ZFunc and RFunc
ZFunc <- function(x) {
  (x - 0.2) / 0.4 + sin(2 * pi * x) * 0.1  # Adding a small oscillatory component
}

RFunc <- function(x) {
  ((ZFunc(x))^3 - 2 * ZFunc(x)) / sqrt(6) + cos(3 * pi * x) * 0.2  # Adding cos to introduce more non-linearity
}

# Multidimensional Data Generating Process function (with K covariates)
generate_multivariate_data <- function(N, K, L) {
  # Generate N observations for K covariates, values between 1 and L
  X <- matrix(sample(1:L, N * K, replace = TRUE), ncol = K)
  
  # Calculate the response based on vectorized operations
  X1 <- X[, 1]
  X2 <- X[, 2]
  X3 <- X[, 3]
  
  # Compute the response Y using vectorized operations
  Y <- 1 - X1 + X1^3 + 2 * RFunc(X1)  # X1's influence with non-linear and cubic terms
  Y <- Y + (X2^4) * sin(X3 * pi / 2)  # Interaction term causing gradient variations between X2 and X3
  Y <- Y + (X2^2) * log(X1)  # Introduce a logarithmic term to cause varying slopes for X1
  
  # Add random noise to the response
  Y <- Y + rnorm(N, mean = 0, sd = 0.5)  # Adding random noise
  
  # Create a dataframe with the results
  data <- data.frame(X1 = X1, X2 = X2, X3 = X3, Y = Y)
  return(data)
}

# Generate data with 1000 samples, 3 covariates, and 5 discrete levels
set.seed(42)
data <- generate_multivariate_data(N = 1000, K = 2, L = 5)

# Create a 3D surface plot of the data
fig <- plot_ly(data, x = ~X1, y = ~X2, z = ~Y) %>%
  add_markers() %>%
  layout(
    scene = list(
      xaxis = list(title = "X1"),
      yaxis = list(title = "X2"),
      zaxis = list(title = "Y")
    ),
    title = "Multidimensional Data with Changing Gradients"
  )

# Show the plot
fig



### GGPlot ###
ggplot() + geom_function(fun = function(x){1-x+x^2 + 2*RFunc(x) +x^3 * sin(x)*pi/2})
