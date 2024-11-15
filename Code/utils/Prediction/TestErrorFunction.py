# Summary: Calculates the RMSE of the test set.
# Input:
#   df_Test: The test data.
# Output:
# RMSE: The residual mean squared error of the predicted values and their true values in the test set. 

### Libraries ###
import numpy as np

### Function ###
def TestErrorFunction(InputModel, df_Test):
    Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
    RMSE = np.mean((Prediction - df_Test["Y"])**2)
    return(RMSE)