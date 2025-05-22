# Summary: Initializes and fits a ridge regression model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   regularization: Ridge regression penalty (alpha_val is mapped to regularization).
# Output: RidgeRegressionModel: A ridge regression model.

### Libraries ###
from sklearn.linear_model import Ridge

### Function ###
def RidgeRegressionFunction(X_train_df, y_train_series, regularization, **kwargs):
    RidgeRegressionModel = Ridge(alpha=regularization)
    RidgeRegressionModel.fit(X_train_df, y_train_series)
    return RidgeRegressionModel