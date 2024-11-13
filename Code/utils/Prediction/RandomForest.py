### Libraries ###
from sklearn.ensemble import RandomForestRegressor

### Function ###
def RandomForestRegressorFunction(df_Train, n_estimators, Seed):
    RandomForestModel = RandomForestRegressor(n_estimators=n_estimators, random_state=Seed)
    RandomForestModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return RandomForestModel
