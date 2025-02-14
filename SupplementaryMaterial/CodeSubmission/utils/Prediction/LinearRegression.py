# Summary: Initializes and fits a linear regression model.
# Input:
#   df_Train: The training data.
# Output:
# LinearRegressionModel: A linear regression model.

### Libraries ###
from sklearn.linear_model import LinearRegression

### Function ###
def LinearRegressionFunction(df_Train):
    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return LinearRegressionModel