# utils/Selector/GreedySamplingSelector.py

# Summary: Implements the greedy sampling methods from Wu, Lin, and Huang (2018).
#          GSx samples based on the covariate space, GSy based on the output space, and iGS on both.

### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target # Ensure this is imported

class GreedySamplingSelector:
    def __init__(self, strategy: str, distance: str = "euclidean", BatchSize: int = 1, Seed: int = None, **kwargs):
        """
        Initializes the GreedySamplingSelector.

        Args:
            strategy (str): The greedy sampling strategy to use. Must be one of {'GSx', 'GSy', 'iGS'}.
            distance (str, optional): The distance metric to use (e.g., 'euclidean'). Defaults to "euclidean".
            BatchSize (int, optional): The number of observations to recommend. Defaults to 1 for greedy sampling.
                                       Note: Greedy sampling typically selects one by one, batching needs
                                       iterative application or specific batch-greedy algorithms.
            Seed (int, optional): Seed for reproducibility. (Not directly used by core logic here, but for consistency).
            **kwargs: Catches any other arguments not used by this selector's init.
        """
        if strategy not in ['GSx', 'GSy', 'iGS']:
            raise ValueError(f"Invalid greedy sampling strategy: {strategy}. Must be 'GSx', 'GSy', or 'iGS'.")
        self.strategy = strategy
        self.distance = distance
        self.BatchSize = BatchSize # Greedy sampling typically picks 1. If BatchSize > 1, this needs adaptation.
        self.Seed = Seed # Stored for consistency

    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, auxiliary_columns: list = None) -> dict:
        """
        Selects observations from the candidate set using the specified greedy sampling strategy.

        Args:
            df_Candidate (pd.DataFrame): The candidate set from which to select observations.
            Model: The trained predictive model instance (required for 'GSy' and 'iGS').
            df_Train (pd.DataFrame): The current training set (required for all strategies).
            auxiliary_columns (list, optional): A list of column names that are not features.

        Returns:
            dict: Contains "IndexRecommendation" (List[int]) of the selected observations.
        """
        if df_Train is None:
            raise ValueError("GreedySamplingSelector requires df_Train to be provided for distance calculations.")
        if self.strategy in ['GSy', 'iGS'] and Model is None:
            raise ValueError(f"Strategy '{self.strategy}' requires a 'Model' to be provided.")
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # Prepare candidate and training features, excluding auxiliary columns
        X_Candidate, _ = get_features_and_target(
            df=df_Candidate, target_column_name="Y", auxiliary_columns=auxiliary_columns
        )
        X_Train, y_Train = get_features_and_target(
            df=df_Train, target_column_name="Y", auxiliary_columns=auxiliary_columns
        )

        # Ensure X_Candidate and X_Train are NumPy arrays for cdist
        X_Candidate_np = X_Candidate.values
        X_Train_np = X_Train.values

        # --- GSx Logic ---
        d_nX = None
        if self.strategy in ['GSx', 'iGS']:
            d_nmX = cdist(X_Candidate_np, X_Train_np, metric=self.distance)
            d_nX = d_nmX.min(axis=1) # Min distance from each candidate to any train point

        # --- GSy Logic ---
        d_nY = None
        if self.strategy in ['GSy', 'iGS']:
            # Ensure Model.predict takes DataFrame, or convert X_Candidate_np back to DataFrame for predict
            # Based on our refactoring, Model.predict expects a DataFrame
            Predictions = Model.predict(X_Candidate) # Pass DataFrame to predictor.predict()

            # y_Train is a Series, need to convert to 2D array for cdist
            d_nmY = cdist(Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric=self.distance)
            d_nY = d_nmY.min(axis=1) # Min distance from each predicted candidate output to any train output

        # --- Select based on strategy ---
        MaxRowNumber = -1
        if self.strategy == 'GSx':
            MaxRowNumber = np.argmax(d_nX)
        elif self.strategy == 'GSy':
            MaxRowNumber = np.argmax(d_nY)
        elif self.strategy == 'iGS':
            if d_nX is None or d_nY is None:
                raise RuntimeError("iGS strategy requires both GSx and GSy components, but one was not computed.")
            d_nXY = d_nX * d_nY
            MaxRowNumber = np.argmax(d_nXY)

        # Return the index of the selected observation(s)
        # Greedy Sampling typically selects one observation. If BatchSize > 1,
        # you'd need an iterative selection process here, where you select one,
        # add it to a temporary batch, remove it from candidates, and re-run the loop.
        # For now, it returns only one observation.
        IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

        # The selector's output always expects a list of indices, even if it's just one.
        return {"IndexRecommendation": [float(IndexRecommendation)]}