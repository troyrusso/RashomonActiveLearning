### Libraries ###
from treefarms import TREEFARMS as tf
import pandas as pd
import numpy as np
import torch # Needed for predict_proba_K output consistency
from scipy import stats # For mode calculation in predict
import warnings # For warnings if no trees found

DEFAULT_STATIC_CONFIG = {
    "depth_budget": 3,
    "rashomon_ignore_trivial_extensions": True,
}

### Predict Single Tree ##
def _predict_single_tree(tree_model, X_df: pd.DataFrame):
    predictions = []
    data = X_df.values 
    for i in range(data.shape[0]):
        prediction, _ = tree_model.classify(data[i, :])
        predictions.append(prediction)
    return pd.Series(predictions, index=X_df.index)

### Score Single Tree ##
def _score_single_tree(tree_model, X: pd.DataFrame, y: pd.Series):
    return (_predict_single_tree(tree_model, X) == y).mean()

### LFR Predictor ###
class TreefarmsLFRPredictor:

    ### Initialize Model ###
    def __init__(self, regularization: float, RashomonThreshold: float,
                 RashomonThresholdType: str = "Adder", Seed: int = None, **kwargs):
        self.regularization = regularization
        self.full_epsilon = RashomonThreshold 
        self.epsilon = RashomonThreshold 
        self.RashomonThresholdType = RashomonThresholdType
        self.Seed = Seed
        self.static_config = DEFAULT_STATIC_CONFIG.copy()
        self.static_config["regularization"] = self.regularization
        self.tf = None 
        self.all_trees = [] 
        self.trees_in_scope = [] 
        self.X_train_current = pd.DataFrame() 
        self.y_train_current = pd.Series() 

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):

        ## Update current cumulative training data ##
        self.X_train_current = X_train_df.copy()
        self.y_train_current = y_train_series.copy()

        ## Configure TreeFarms for a full fit ##
        config = self.static_config.copy()
        if self.RashomonThresholdType == "Adder":
            config['rashomon_bound_adder'] = self.full_epsilon 
        elif self.RashomonThresholdType == "Multiplier":
            config['rashomon_bound_multiplier'] = self.full_epsilon
        else:
            warnings.warn(f"Unsupported RashomonThresholdType: {self.RashomonThresholdType}. Defaulting to 'Adder' logic.")
            config['rashomon_bound_adder'] = self.full_epsilon

        ## Refit TREEFARMS ##
        self.tf = tf(config)
        self.tf.fit(self.X_train_current, self.y_train_current)

        ## Store all trees ##
        self.all_trees = [self.tf[i] for i in range(self.tf.get_tree_count())]
        self.trees_in_scope = self.all_trees.copy() 
        self.epsilon = self.full_epsilon 

    ### Refit ###
    def refit(self, X_to_add: pd.DataFrame, y_to_add: pd.Series, epsilon: float):
        
        ## Concatenate new data with existing cumulative training data ##
        all_X = pd.concat([self.X_train_current, X_to_add], ignore_index=True)
        all_y = pd.concat([self.y_train_current, y_to_add], ignore_index=True)

        ## Update the cumulative training data stored internally ##
        self.X_train_current = all_X
        self.y_train_current = all_y

        ## Set epsilon ##
        self.epsilon = epsilon

        ## LFR Logic: Decide between full refit and subsetting based on epsilon change ##
        if self.full_epsilon < self.epsilon:
            self.full_epsilon = self.epsilon 
            self.fit(self.X_train_current, self.y_train_current) 
        else:
            objectives = np.array([_score_single_tree(tree, self.X_train_current, self.y_train_current) for tree in self.all_trees])
            errors = 1 - objectives 
            min_error = np.min(errors)
            self.trees_in_scope = [self.all_trees[i] for i, err in enumerate(errors) if err <= min_error + self.epsilon]
            if not self.trees_in_scope: 
                self.trees_in_scope = self.all_trees.copy() 

    ### Helper to get predictions from all trees currently in scope ###
    def _get_ensemble_predictions_df(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        if not self.trees_in_scope:
            warnings.warn("No trees currently in scope for ensemble predictions. Returning empty DataFrame.")
            return pd.DataFrame(index=X_data_df.index)

        predictions_list = [_predict_single_tree(tree, X_data_df) for tree in self.trees_in_scope]
        ensemble_predictions_df = pd.concat(predictions_list, axis=1)
        ensemble_predictions_df.columns = [f"Tree_{i}" for i in range(ensemble_predictions_df.shape[1])]
        return ensemble_predictions_df


    ### Predict Model (Ensemble Mode Prediction) ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        if self.tf is None: 
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if not self.trees_in_scope: 
             warnings.warn("No trees in scope for prediction. Returning zeros.")
             return np.zeros(X_data_df.shape[0], dtype=int) 
        ensemble_predictions_df = self._get_ensemble_predictions_df(X_data_df)
        mode_predictions = stats.mode(ensemble_predictions_df, axis=1)[0].squeeze()
        return mode_predictions.astype(int) 

    ### Get prediction probabilities ###
    def predict_proba_K(self, X_data_np: np.ndarray, K_samples: int = None) -> torch.Tensor:
        if not self.trees_in_scope: 
            num_samples = X_data_np.shape[0]
            num_classes = len(np.unique(self.y_train_current)) if self.y_train_current is not None else 2 # Default to 2 classes
            warnings.warn("No trees in scope for predict_proba_K. Returning default tensor.")
            return torch.full((num_samples, 0, num_classes), -float('inf'), dtype=torch.float32)

        X_data_df = pd.DataFrame(X_data_np, columns=self.X_train_current.columns) # Use stored train columns
        ensemble_predictions_df = self._get_ensemble_predictions_df(X_data_df)
        
        num_samples = ensemble_predictions_df.shape[0]
        num_trees_in_ensemble = ensemble_predictions_df.shape[1]
        
        unique_classes_sorted = np.sort(np.unique(self.y_train_current)) # Ensure order for mapping
        num_classes = len(unique_classes_sorted)
        class_mapping = {cls: i for i, cls in enumerate(unique_classes_sorted)}
        
        log_probs_N_K_C = torch.full((num_samples, num_trees_in_ensemble, num_classes), -float('inf'), dtype=torch.float32)

        for sample_idx in range(num_samples):
            for tree_idx in range(num_trees_in_ensemble):
                predicted_class_label = ensemble_predictions_df.iloc[sample_idx, tree_idx]
                mapped_class_index = class_mapping.get(predicted_class_label)
                if mapped_class_index is not None:
                    log_probs_N_K_C[sample_idx, tree_idx, mapped_class_index] = 0.0

        return log_probs_N_K_C

    ### Get total and unique tree counts ###
    def get_tree_counts(self) -> dict:
        if self.tf is None:
            return {"AllTreeCount": 0, "UniqueTreeCount": 0}

        all_tree_count = self.tf.get_tree_count() 
        unique_tree_count = len(self.trees_in_scope)

        return {"AllTreeCount": all_tree_count, "UniqueTreeCount": unique_tree_count}
    
    ### Get predictions from each individual tree ###
    def get_raw_ensemble_predictions(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        return self._get_ensemble_predictions_df(X_data_df)