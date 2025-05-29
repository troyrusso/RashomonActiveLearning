# utils/Prediction/LinearRegressionPredictor.py

# Summary: Initializes and fits a linear regression model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
# Output:
# LinearRegressionModel: A linear regression model instance.

### Libraries ###
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression

class LinearRegressionPredictor:
    def __init__(self, **kwargs):
        """
        Initializes the LinearRegressionPredictor.

        Args:
            **kwargs: Allows for additional arguments, but they are not used by this model.
        """
        self.model = None

    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        """
        Fits the Linear Regression model.

        Args:
            X_train_df (pd.DataFrame): The training features.
            y_train_series (pd.Series): The training target.
        """
        self.model = LinearRegression()
        self.model.fit(X_train_df, y_train_series)

    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            X_data_df (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: Predicted regression values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self.model.predict(X_data_df)