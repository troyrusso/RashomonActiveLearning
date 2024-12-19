# Summary: Calculates the loss (RMSE for regression and classification error for classification) of the test set.
# Input:
#   InputModel: The prediction model used.
#   df_Test: The test data.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
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
        ErrorVal = [ErrorVal.tolist()]

    ### Classification Error ###
    if(Type == "Classification"):

        # Rashomon Classification #
        if 'TREEFARMS' in str(type(InputModel)):
            ErrorVal =[]
            for i in range(InputModel.get_tree_count()):
                ModelError = InputModel[i].error(df_Test.loc[:, df_Test.columns != "Y"], df_Test["Y"])
                ErrorVal.append(ModelError)
        else:
            Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
            ErrorVal = np.mean(Prediction != df_Test["Y"])
            ErrorVal = [ErrorVal.tolist()]

    ### Return ###
    return ErrorVal
            