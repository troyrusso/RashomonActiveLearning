### Libraries ###
import math
import torch
import numpy as np
from typing import List
# from tqdm.auto import tqdm 
from dataclasses import dataclass
from utils.Auxiliary.DataFrameUtils import get_features_and_target # Import the new function


### Toma ###
# This mock class mimics the structure expected by the decorator @toma.execute.chunked
# In a production environment, you would 'pip install toma' and 'import toma'
class TomaExecute:
    def chunked(self, data, chunk_size):
        def decorator(func):
            def wrapper(*args, **kwargs):
                N = data.shape[0]
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    func(data[start:end], start, end)
            return wrapper
        return decorator
class Toma:
    def __init__(self):
        self.execute = TomaExecute()
toma = Toma()

### Conditional Entropy Function ###
def ComputeConditionalEntropyFunction(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Computes the Expected Conditional Entropy (E_theta[H(y|x,theta)]) for each data point.
    """
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    # pbar = tqdm(total=N, desc="Conditional Entropy", leave=False) # Initialize pbar

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        EntropyVals = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
        entropies_N[start:end].copy_(-torch.sum(EntropyVals, dim=(1, 2)) / K)
        # pbar.update(end - start) # Update pbar inside the compute function
    # pbar.close() # Close pbar after all chunks are processed

    return entropies_N

### Entropy Function ###
def ComputeEntropyFunction(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Computes the Marginal Entropy (H(y|x,D)) for each data point.
    """
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    # pbar = tqdm(total=N, desc="Entropy", leave=False) # Initialize pbar

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        # pbar.update(end - start) # Update pbar inside the compute function
    # pbar.close() # Close pbar after all chunks are processed

    return entropies_N


### Data Class for Output ###
@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


### BALD Selector Function ###
def BaldSelectorFunction(Model, df_Candidate, BatchSize, K_BALD_Samples=20, auxiliary_columns = None):
    """
    Selects the most informative observations using the BALD (Bayesian Active Learning by Disagreement) criterion.

    Args:
        Model: The trained Bayesian Neural Network (or similar) model with a .predict_proba_K method.
        df_Candidate (pd.DataFrame): The DataFrame of unlabeled candidate observations.
        BatchSize (int): The number of observations to recommend.
        K_BALD_Samples (int): The number of samples to draw from the model's posterior
                              for BALD calculation (K in log_probs_N_K_C).

    Returns:
        CandidateBatch: A dataclass containing the BALD scores and original DataFrame indices
                        of the recommended observations.
    """

    # Extract features from df_Candidate (assuming 'Y' is the label column if present)
    # Ensure X_candidate_np is a NumPy array for Model.predict_proba_K
    X_candidate_df, _ = get_features_and_target( # No need for target here, so _
            df=df_Candidate,
            target_column_name="Y", # Still need this to identify the Y column
            auxiliary_columns=auxiliary_columns # Use the passed aux cols
        )
    X_candidate_np = X_candidate_df.values

    # Generate log_probs_N_K_C using the BNN's prediction method
    # This tensor will have shape (N_candidate_obs, K_BALD_Samples, num_classes)
    log_probs_N_K_C = Model.predict_proba_K(X_candidate_np, K_BALD_Samples)

    # Determine dimensions
    N_candidate, K, C = log_probs_N_K_C.shape
    batch_size_actual = min(BatchSize, N_candidate)

    # Compute Uncertainty Metrics (BALD scores)
    # BALD Score = H(y|x,D) - E_theta[H(y|x,theta)]
    EnsembleEntropy = ComputeEntropyFunction(log_probs_N_K_C)
    ConditionalEntropy = ComputeConditionalEntropyFunction(log_probs_N_K_C)
    UncertaintyMetrics = EnsembleEntropy - ConditionalEntropy

    # Get the top BatchSize observations based on BALD scores
    # torch.topk returns values and indices
    top_scores, top_local_indices = torch.topk(UncertaintyMetrics, batch_size_actual)

    # Convert the local indices (within the log_probs_N_K_C tensor)
    # back to the original DataFrame indices of df_Candidate
    candidate_df_indices = df_Candidate.index.values # This gets the actual pandas index as a NumPy array
    IndexRecommendation = candidate_df_indices[top_local_indices.cpu().numpy().astype(int)].tolist()

    # Return the selected batch
    Output = {"IndexRecommendation": IndexRecommendation}
    return Output