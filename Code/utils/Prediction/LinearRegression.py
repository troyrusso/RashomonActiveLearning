# Summary: Initializes and fits a linear regression model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
# Output:
# LinearRegressionModel: A linear regression model.

### Libraries ###
from sklearn.linear_model import LinearRegression

### Function ###
def LinearRegressionFunction(X_train_df, y_train_series, **kwargs):
    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.fit(X_train_df, y_train_series)
    return LinearRegressionModel