### Libraries ###
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierPredictor:
    def __init__(self, Seed: int, n_estimators: int = 100, **kwargs):
        self.Seed = Seed
        self.n_estimators = n_estimators
        self.model = None 
        np.random.seed(self.Seed) 

    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.Seed)
        self.model.fit(X_train_df, y_train_series)

    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self.model.predict(X_data_df)

    def predict_proba(self, X_data_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self.model.predict_proba(X_data_df)

    def get_raw_ensemble_predictions(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns predictions from all individual trees (estimators) in the Random Forest.
        Useful for ensemble-based selectors like QBC.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        individual_tree_preds = []
        
        # Convert X_data_df to a NumPy array here to prevent the UserWarning
        X_data_np = X_data_df.values 
        
        for tree_estimator in self.model.estimators_:
            # Pass the NumPy array to the individual tree's predict method
            individual_tree_preds.append(tree_estimator.predict(X_data_np))
        
        ensemble_predictions_df = pd.DataFrame(np.vstack(individual_tree_preds)).T 
        ensemble_predictions_df.columns = [f"tree_{i}" for i in range(ensemble_predictions_df.shape[1])]
        ensemble_predictions_df.index = X_data_df.index 
        return ensemble_predictions_df