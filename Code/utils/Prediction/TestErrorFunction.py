# Summary: Calculates the RMSE of the test set.
# Input:
#   df_Test: The test data.
# Output:
# RMSE: The residual mean squared error of the predicted values and their true values in the test set. 

### Libraries ###
import numpy as np

### Function ###
def TestErrorFunction(InputModel, df_Test, ModelArgs):

    ### RMSE ###
    if(ModelArgs["Type"] == "Regression"):
        Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
        ErrorVal = np.mean((Prediction - df_Test["Y"])**2)

    ### Classification Error ###
    if(ModelArgs["Type"] == "Classification"):

        # Rashomon Classification #
        if "TopCModels" in ModelArgs.keys():
            ErrorVal = [1-InputModel[i].score(df_Test.loc[:, df_Test.columns != "Y"], df_Test["Y"]) for i in range(InputModel.get_tree_count())]

        elif not("TopCModels" in ModelArgs.keys()):
            ErrorVal = np.mean(Prediction != df_Test["Y"])


    return ErrorVal
            