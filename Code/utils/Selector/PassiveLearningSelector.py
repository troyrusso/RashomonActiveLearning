# utils/Selector/PassiveLearningSelector.py

# Summary: Chooses an index at random from the candidate set to be queried.

### Libraries ###
import pandas as pd
import numpy as np

class PassiveLearningSelector:
    def __init__(self, BatchSize: int = 5, Seed: int = None, **kwargs):
        """
        Initializes the PassiveLearningSelector.

        Args:
            BatchSize (int, optional): The number of observations to recommend. Defaults to 5.
            Seed (int, optional): Seed for reproducibility of random sampling. Defaults to None.
            **kwargs: Catches any other arguments not used by this selector's init.
        """
        self.BatchSize = BatchSize
        self.Seed = Seed
        
        # Set seed for reproducibility of random sampling
        if self.Seed is not None:
            np.random.seed(self.Seed)

    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, auxiliary_columns: list = None) -> dict:
        """
        Selects observations randomly from the candidate set.

        Args:
            df_Candidate (pd.DataFrame): The candidate set.
            Model: The predictive model instance (not used by PassiveLearning, but kept for consistent interface).
            df_Train (pd.DataFrame, optional): The current training set (not used by PassiveLearning, but kept for consistent interface). Defaults to None.
            auxiliary_columns (list, optional): A list of column names that are not features (not used by PassiveLearning, but kept for consistent interface).

        Returns:
            dict: Contains "IndexRecommendation" (List[int]) of the selected observations.
        """
        if df_Candidate.shape[0] >= self.BatchSize:
             QueryObservation = df_Candidate.sample(n=self.BatchSize, random_state=self.Seed) 
             IndexRecommendation = list(QueryObservation.index)
        else:
            IndexRecommendation = list(df_Candidate.index) # Select all if fewer than BatchSize

        return {"IndexRecommendation": IndexRecommendation}