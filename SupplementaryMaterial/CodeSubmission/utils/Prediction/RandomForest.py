# Summary: Initializes and fits a random forest regressor/classifier model.
# Input:
#   df_Train: The training data.
#   n_estimators: The number of trees for a random forest.
#   Seed: Seed for reproducability.
# Output:
# RandomForestModel: A random forest regressor/classifier model.

### Libraries ###
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

### Function ###
def RandomForestRegressorFunction(df_Train, n_estimators, Seed):
    RandomForestRegressorModel = RandomForestRegressor(n_estimators=n_estimators, random_state=Seed)
    RandomForestRegressorModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return RandomForestRegressorModel

### Function ###
def RandomForestClassificationFunction(df_Train, n_estimators, Seed):
    RandomForestClassificationModel = RandomForestClassifier(n_estimators=n_estimators, random_state=Seed)
    RandomForestClassificationModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return RandomForestClassificationModel

