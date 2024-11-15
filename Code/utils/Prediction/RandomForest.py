# Summary: Initializes and fits a random forest regressor model.
# Input:
#   df_Train: The training data.
#   n_estimators: The number of trees for each forest.
#   Seed: Seed for reproducability.
# Output:
# RandomForestModel: A random forest regressor model.


### Libraries ###
from sklearn.ensemble import RandomForestRegressor

### Function ###
def RandomForestRegressorFunction(df_Train, n_estimators, Seed):
    RandomForestModel = RandomForestRegressor(n_estimators=n_estimators, random_state=Seed)
    RandomForestModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return RandomForestModel
