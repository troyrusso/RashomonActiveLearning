# Summary: Initializes and fits a random forest regressor/classifier model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   n_estimators: The number of trees for a random forest.
#   Seed: Seed for reproducibility.
# Output:
# RandomForestModel: A random forest regressor/classifier model.


### Libraries ###
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

### Regression ###
def RandomForestRegressorFunction(X_train_df, y_train_series, n_estimators, Seed, **kwargs):
    RandomForestRegressorModel = RandomForestRegressor(n_estimators=n_estimators, random_state=Seed)
    RandomForestRegressorModel.fit(X_train_df, y_train_series)
    return RandomForestRegressorModel

### Classification ###
def RandomForestClassificationFunction(X_train_df, y_train_series, n_estimators, Seed, **kwargs):
    RandomForestClassificationModel = RandomForestClassifier(n_estimators=n_estimators, random_state=Seed)
    RandomForestClassificationModel.fit(X_train_df, y_train_series)
    return RandomForestClassificationModel