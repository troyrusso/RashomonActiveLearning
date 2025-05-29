# Summary: Initializes and fits a random forest regressor model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   n_estimators: The number of trees for a random forest.
#   Seed: Seed for reproducibility.
# Output:
# RandomForestRegressorModel: A random forest regressor model instance.

### Libraries ###
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestRegressorPredictor:
    def __init__(self, Seed: int, n_estimators: int = 100, **kwargs):
        """
        Initializes the RandomForestRegressorPredictor.

        Args:
            Seed (int): Seed for reproducibility.
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
            **kwargs: Allows for additional arguments, but they are not used by this model.
        """
        self.Seed = Seed
        self.n_estimators = n_estimators
        self.model = None 
        np.random.seed(self.Seed)

    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        """
        Fits the Random Forest Regressor model.

        Args:
            X_train_df (pd.DataFrame): The training features.
            y_train_series (pd.Series): The training target.
        """
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.Seed)
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