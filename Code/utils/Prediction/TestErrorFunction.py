# Summary: Calculates the RMSE of the test set.
# Input:
#   df_Test: The test data.
# Output:
# RMSE: The residual mean squared error of the predicted values and their true values in the test set. 

### Libraries ###
import numpy as np

### Function ###
def TestErrorFunction(InputModel, df_Test, Type):

    ### RMSE ###
    if(Type == "Regression"):
        Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
        ErrorVal = np.mean((Prediction - df_Test["Y"])**2)

    ### Classification Error ###
    if(Type == "Classification"):

        # Rashomon Classification #
        if 'TREEFARMS' in str(type(InputModel)):
            ErrorVal =[]
            for i in range(InputModel.get_tree_count()):
                ModelError = InputModel[i].error(df_Test.loc[:, df_Test.columns != "Y"], df_Test["Y"])
                ErrorVal.append(ModelError)
        else:
            ErrorVal = np.mean(Prediction != df_Test["Y"])

    ### Return ###
    return ErrorVal
            