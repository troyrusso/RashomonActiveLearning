### Libraries ###
import numpy as np

### Function ###
def TestErrorFunction(InputModel, df_Test):
    Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
    RMSE = np.mean((Prediction - df_Test["Y"])**2)
    return(RMSE)