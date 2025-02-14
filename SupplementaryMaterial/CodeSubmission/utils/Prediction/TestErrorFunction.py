# Summary: Calculates the loss (RMSE for regression and classification error for classification) of the test set.
# Input:
#   InputModel: The prediction model used.
#   df_Test: The test data.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output:
# RMSE: The residual mean squared error of the predicted values and their true values in the test set. 

### Libraries ###
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

### Function ###
def TestErrorFunction(InputModel, df_Test, Type):

    ### RMSE ###
    if(Type == "Regression"):
        Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
        ErrorVal = np.mean((Prediction - df_Test["Y"])**2)
        ErrorVal = [ErrorVal.tolist()]
        Output = {"ErrorVal": ErrorVal}

    ### Classification Error ###
    if(Type == "Classification"):
        
        ## Rashomon Classification ##
        if 'TREEFARMS' in str(type(InputModel)):
            TreeCounts = InputModel.get_tree_count()

            # Duplicate #
            PredictionArray_Duplicate = pd.DataFrame(np.array([InputModel[i].predict(df_Test.loc[:, df_Test.columns != "Y"]) for i in range(TreeCounts)]))
            PredictionArray_Duplicate.columns = df_Test.index.astype(str)
            EnsemblePrediction_Duplicate = pd.Series(stats.mode(PredictionArray_Duplicate)[0])
            EnsemblePrediction_Duplicate.index = df_Test["Y"].index
            Error_Duplicate = float(f1_score(df_Test["Y"], EnsemblePrediction_Duplicate, average='micro'))
            # AllTreeCount = PredictionArray_Duplicate.shape[0]

            # Unique #
            PredictionArray_Unique = pd.DataFrame(PredictionArray_Duplicate).drop_duplicates()
            EnsemblePrediction_Unique = pd.Series(stats.mode(PredictionArray_Unique)[0])
            EnsemblePrediction_Unique.index = df_Test["Y"].index
            # Error_Unique = float(f1_score(df_Test["Y"], EnsemblePrediction_Unique, average='micro'))
            # UniqueTreeIndices= PredictionArray_Unique.index
            # UniqueTreeCount = PredictionArray_Unique.shape[0]


            # Output #
            Output = {"Error_Duplicate": Error_Duplicate,
                    #   "Error_Unique": Error_Unique,
                    #   "PredictionArray_Duplicate" : PredictionArray_Duplicate,
                    #   "PredictionArray_Unique" : PredictionArray_Unique,
                    #   "UniqueTreeIndices": UniqueTreeIndices,
                    #   "AllTreeCount": AllTreeCount,
                    #   "UniqueTreeCount": UniqueTreeCount
                      }

        else:
            Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
            # ErrorVal = float(np.mean(Prediction != df_Test["Y"]))
            ErrorVal = float(f1_score(df_Test["Y"], Prediction, average='micro'))
            Output = {"ErrorVal": ErrorVal}

    ### Return ###
    return Output
            