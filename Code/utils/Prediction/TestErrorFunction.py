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
from utils.Auxiliary.DataFrameUtils import get_features_and_target


### Function ###

### Function ###
def TestErrorFunction(InputModel, df_Test, Type, auxiliary_columns=None):

    X_test_df, y_test_series = get_features_and_target(
        df=df_Test,
        target_column_name="Y",
        auxiliary_columns=auxiliary_columns 
    )
    X_test_np = X_test_df.values
    y_test_np = y_test_series.values 
    
    ### RMSE ###
    if Type == "Regression":
        Prediction = InputModel.predict(X_test_df)
        ErrorVal = np.mean((Prediction - y_test_series)**2)
        Output = {"ErrorVal": ErrorVal.tolist()}

    ### Classification Error ###
    elif Type == "Classification":
        
        if hasattr(InputModel, 'predict_proba_K'):
            K_for_test_eval = 100 
            
            # Pass the already correctly filtered X_test_np
            log_probs_N_K_C_test = InputModel.predict_proba_K(X_test_np, K_for_test_eval)

            # Convert log-probabilities to probabilities for ensemble prediction
            probs_N_K_C_test = torch.exp(log_probs_N_K_C_test)

            # Average probabilities across K samples for each observation and class
            mean_probs_N_C_test = torch.mean(probs_N_K_C_test, dim=1)

            # Get the most likely class (predicted label) for each observation
            ensemble_prediction_test = torch.argmax(mean_probs_N_C_test, dim=1).cpu().numpy()

            # Calculate F1 score
            ErrorVal = float(f1_score(y_test_np, ensemble_prediction_test, average='micro'))
            Output = {"ErrorVal": ErrorVal}
            
            # If the model is a TreeFARMS-like model that provides tree counts, include them
            if hasattr(InputModel, 'get_tree_counts'): 
                 tree_counts = InputModel.get_tree_counts() 
                 Output["AllTreeCount"] = tree_counts["AllTreeCount"]
                 Output["UniqueTreeCount"] = tree_counts["UniqueTreeCount"]
        else:
            Prediction = InputModel.predict(X_test_df) 
            ErrorVal = float(f1_score(y_test_np, Prediction, average='micro'))
            Output = {"ErrorVal": ErrorVal}

    ### Return ###
    return Output

            