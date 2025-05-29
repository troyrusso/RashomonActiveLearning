# utils/Prediction/TreeFarmsPredictor.py

# Summary: Initializes and fits a treefarms model using the standard TreeFarms library.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThresholdType: Type of Rashomon threshold calculation ("Adder" or "Multiplier").
#   RashomonThreshold: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss or epsilon value.
# Output:
# treeFarmsModel: A treefarms model.

### Libraries ###
from treeFarms.treefarms.model.treefarms import TREEFARMS
import pandas as pd
import numpy as np # For consistency with other predictor files
from scipy import stats # For mode calculation in predict (needed for classification)
import torch # To make predict_proba_K consistent with BNN output structure

class TreeFarmsPredictor:
    def __init__(self, regularization: float, RashomonThreshold: float, 
                 RashomonThresholdType: str = "Adder", Seed: int = None, **kwargs):
        """
        Initializes the TreeFarmsPredictor.
        """
        self.regularization = regularization
        self.RashomonThreshold = RashomonThreshold
        self.RashomonThresholdType = RashomonThresholdType
        self.Seed = Seed 

        self.model = None 
        self.all_trees = [] 
        self.X_train_columns = None 
        self.y_train_classes = None
 

    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        """
        Fits the TreeFarms model.
        """
        self.X_train_columns = X_train_df.columns 
        self.y_train_classes = y_train_series 
        
        config = {"regularization": self.regularization}

        if self.RashomonThresholdType == "Adder":
            config["rashomon_bound_adder"] = self.RashomonThreshold
        elif self.RashomonThresholdType == "Multiplier":
            config["rashomon_bound_multiplier"] = self.RashomonThreshold
        else:
            raise ValueError(f"Unknown RashomonThresholdType: {self.RashomonThresholdType}")

        self.model = TREEFARMS(config)
        self.model.fit(X_train_df, y_train_series)
        self.all_trees = [self.model[i] for i in range(self.model.get_tree_count())]


    # Helper to predict with a single tree from the ensemble
    def _predict_single_tree(self, tree_model, X_df: pd.DataFrame):
        predictions = []
        data = X_df.values
        for i in range(data.shape[0]):
            prediction, _ = tree_model.classify(data[i, :])
            predictions.append(prediction)
        return pd.Series(predictions, index=X_df.index)

    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        """
        Makes ensemble (mode) predictions on new data.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        if not self.all_trees: 
            return np.zeros(X_data_df.shape[0]) 

        # Get predictions from all trees in the ensemble
        ensemble_predictions_list = [self._predict_single_tree(tree, X_data_df) for tree in self.all_trees]
        ensemble_predictions_df = pd.concat(ensemble_predictions_list, axis=1)
        
        # Calculate mode across rows (axis=1) for each sample
        mode_predictions = stats.mode(ensemble_predictions_df, axis=1)[0].squeeze()
        return mode_predictions.astype(int) 

    def predict_proba_K(self, X_data_np: np.ndarray, K_samples: int = None) -> torch.Tensor:
        """
        Generates K (number of trees in ensemble) "stochastic" log probabilities for BALD.
        Each "sample" is a one-hot prediction from one tree.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # Convert numpy array back to DataFrame for ensemble prediction
        X_data_df = pd.DataFrame(X_data_np, columns=self.X_train_columns)

        ensemble_predictions_list = [self._predict_single_tree(tree, X_data_df) for tree in self.all_trees]
        if not ensemble_predictions_list:
            raise RuntimeError("No trees found in TreeFarms ensemble for predict_proba_K.")
            
        ensemble_predictions_df = pd.concat(ensemble_predictions_list, axis=1)

        num_samples = ensemble_predictions_df.shape[0]
        num_trees_in_ensemble = ensemble_predictions_df.shape[1]
        
        # Determine unique classes from training data
        unique_classes = np.unique(self.y_train_classes) 
        num_classes = len(unique_classes)
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        
        # Initialize tensor for log probabilities
        # Shape: (N_samples, K_trees_in_ensemble, num_classes)
        log_probs_N_K_C = torch.full((num_samples, num_trees_in_ensemble, num_classes), -float('inf'), dtype=torch.float32)

        for sample_idx in range(num_samples):
            for tree_idx in range(num_trees_in_ensemble):
                predicted_class_label = ensemble_predictions_df.iloc[sample_idx, tree_idx]
                mapped_class_index = class_mapping.get(predicted_class_label)
                if mapped_class_index is not None: # Ensure class is known
                    log_probs_N_K_C[sample_idx, tree_idx, mapped_class_index] = 0.0 # log(1) for the predicted class

        return log_probs_N_K_C

    def get_tree_counts(self) -> dict:
        """
        Returns the count of all trees in the Rashomon set.
        (For this non-LFR version, all_trees is always the full Rashomon set)
        """
        all_tree_count = self.model.get_tree_count() if self.model else 0
        unique_tree_count = self.model.get_unique_tree_count() if self.model and hasattr(self.model, 'get_unique_tree_count') else all_tree_count # Assuming TREEFARMS might have this method
        
        return {"AllTreeCount": all_tree_count, "UniqueTreeCount": unique_tree_count}
    
    def get_raw_ensemble_predictions(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns predictions from all individual trees in the ensemble.
        Useful for ensemble-based selectors like QBC.
        """
        if self.model is None or not self.all_trees:
            raise RuntimeError("Model has not been fitted yet or no trees found. Call .fit() first.")
        
        ensemble_predictions_list = [self._predict_single_tree(tree, X_data_df) for tree in self.all_trees]
        ensemble_predictions_df = pd.concat(ensemble_predictions_list, axis=1)
        ensemble_predictions_df.columns = [f"tree_{i}" for i in range(ensemble_predictions_df.shape[1])]
        return ensemble_predictions_df