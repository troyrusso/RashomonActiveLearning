# Summary: Initializes and fits a ridge regression model.
# Input:
#   df_Train: The training data.
#   alpha_val: Ridge regression penalty.
# Output:
# RidgeRegressionModel: A ridge regression model.

### Libraries ###
from sklearn.linear_model import Ridge

### Function ###
def RidgeRegressionFunction(df_Train, alpha_val):
    RidgeRegressionModel = Ridge(alpha = alpha_val)
    RidgeRegressionModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return RidgeRegressionModel