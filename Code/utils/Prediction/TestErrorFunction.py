# Summary: Calculates the loss (RMSE for regression and classification error for classification) of the test set.
# Input:
#   InputModel: The prediction model used.
#   df_Test: The test data.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output:
# RMSE: The residual mean squared error of the predicted values and their true values in the test set. 

### Libraries ###
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from utils.Auxiliary.DataFrameUtils import get_features_and_target # Import the new function

### Function ###
def TestErrorFunction(InputModel, df_Test, Type, auxiliary_columns=None):

    ### RMSE ###
    if(Type == "Regression"):
        Prediction = InputModel.predict(df_Test.loc[:, df_Test.columns != "Y"])
        ErrorVal = np.mean((Prediction - df_Test["Y"])**2)
        ErrorVal = ErrorVal.tolist()
        Output = {"ErrorVal": ErrorVal}

    ### Classification Error ###
    if(Type == "Classification"):

        X_test_df, y_test_series = get_features_and_target(
            df=df_Test,
            target_column_name="Y",
            auxiliary_columns=auxiliary_columns 
        )
        X_test_np = X_test_df.values
        y_test_np = y_test_series.values 
        
        ## Rashomon Classification ##
        if 'TREEFARMS' in str(type(InputModel)):
            TreeCounts = InputModel.get_tree_count()

            # Duplicate #
            PredictionArray_Duplicate = pd.DataFrame(np.array([InputModel[i].predict(X_test_df) for i in range(TreeCounts)]))
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
            
        elif 'BayesianNeuralNetwork' in str(type(InputModel)):
            K_BALD_for_test_eval = 100

            # Pass the already correctly filtered X_test_np
            log_probs_N_K_C_test = InputModel.predict_proba_K(X_test_np, K_BALD_for_test_eval)

            # Convert log-probabilities to probabilities for ensemble prediction
            probs_N_K_C_test = torch.exp(log_probs_N_K_C_test)

            # Average probabilities across K samples for each observation and class
            mean_probs_N_C_test = torch.mean(probs_N_K_C_test, dim=1)

            # Get the most likely class (predicted label) for each observation
            ensemble_prediction_test = torch.argmax(mean_probs_N_C_test, dim=1).cpu().numpy()

            # Calculate F1 score
            ErrorVal = float(f1_score(y_test_np, ensemble_prediction_test, average='micro'))
            Output = {"ErrorVal": ErrorVal}

        else:
            Prediction = InputModel.predict(X_test_df)
            ErrorVal = float(f1_score(y_test_np, Prediction, average='micro'))
            Output = {"ErrorVal": ErrorVal}

    ### Return ###
    return Output
            