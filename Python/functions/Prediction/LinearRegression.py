### Libraries ###
from sklearn.linear_model import LinearRegression

### Function ###
def LinearRegressionFunction(df_Train):
    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    return LinearRegressionModel