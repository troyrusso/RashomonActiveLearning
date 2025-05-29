# utils/Selector/BatchQBCDiversityFunction.py

# Summary: Query-by-committee function for either random forest or Rashomon's TreeFarms that 
#          recommends an observation from the candidate set to be queried.
# Input:
#   Model: The predictive model instance (e.g., TreeFarmsPredictor, RandomForestClassifierPredictor).
#   df_Candidate: The candidate set.
#   df_Train: The training set.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFarms.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

# NOTE: Incorporate covariate GSx in selection criteria? Good for tie breakers.

### Libraries ###
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from utils.Auxiliary.DataFrameUtils import get_features_and_target # Ensure this is imported

### Function ###
def BatchQBCDiversityFunction(Model, df_Candidate, df_Train, UniqueErrorsInput, DiversityWeight, DensityWeight, BatchSize, auxiliary_columns=None):

    ### Ignore warning (taken care of) ###
    warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in multiply", category=RuntimeWarning)

    # Prepare candidate features
    X_Candidate_df, _ = get_features_and_target(
        df=df_Candidate,
        target_column_name="Y", # Still needed to identify Y column if present
        auxiliary_columns=auxiliary_columns
    )

    ### Predicted Values (Committee Votes) ###
    # This block now uses the standardized 'get_raw_ensemble_predictions' method
    if hasattr(Model, 'get_raw_ensemble_predictions'):
        PredictedValues_df = Model.get_raw_ensemble_predictions(X_Candidate_df)
        PredictedValues = PredictedValues_df.values # Convert to numpy array for vote entropy calculation

        # Retrieve tree counts if the model provides them (like TreeFarmsPredictor)
        Output = {}
        if hasattr(Model, 'get_tree_counts'):
            tree_counts = Model.get_tree_counts()
            Output["AllTreeCount"] = tree_counts.get("AllTreeCount", 0)
            Output["UniqueTreeCount"] = tree_counts.get("UniqueTreeCount", 0)
        else: # For models like RF, which may not have native "unique" tree count
            Output["AllTreeCount"] = PredictedValues_df.shape[1] # Number of estimators
            Output["UniqueTreeCount"] = PredictedValues_df.drop_duplicates().shape[1] # Unique prediction patterns from estimators (if desired)

        # UniqueErrorsInput logic for TreeFarms (now applied to the output of get_raw_ensemble_predictions)
        # UniqueErrorsInput for TreeFarms was about pruning duplicate trees before calculating error.
        # If it refers to reducing the "committee" for QBC, this logic needs thought.
        # For now, if UniqueErrorsInput is true, we consider only unique prediction patterns as the committee.
        if UniqueErrorsInput and hasattr(Model, 'get_tree_counts'): # Only applies to TreeFarms-like models
            # This is a bit of a reinterpretation of UniqueErrorsInput
            # The intent was to count unique error patterns for TreeFarms, not necessarily use unique trees for QBC.
            # If `UniqueErrorsInput` means using only unique committee members for Vote Entropy:
            PredictedValues_df_unique = PredictedValues_df.T.drop_duplicates().T # Get unique columns (unique trees)
            PredictedValues = PredictedValues_df_unique.values
            # The tree counts in Output should reflect this if needed.

    else:
        # For non-ensemble models, or models without 'get_raw_ensemble_predictions'
        # QBC usually implies an ensemble. If you use a single model here,
        # its predictions would be repeated, leading to 0 vote entropy.
        # Raise an error or handle specifically if single-model QBC is desired.
        raise TypeError(f"Model type {type(Model).__name__} does not have 'get_raw_ensemble_predictions' "
                        f"and is not supported for BatchQBCDiversityFunction's current implementation.")

    if PredictedValues is None:
        raise RuntimeError("PredictedValues could not be obtained for BatchQBCDiversityFunction.")


    ### Vote Entropy ###
    VoteC = {}
    LogVoteC = {}
    VoteEntropy = {}
    # Use unique classes from the full training set or overall dataset for comprehensive class list
    # df_Train contains the training data (features and target), which should have all possible classes seen so far.
    # If using a specific dataset, all possible classes might need to be known.
    UniqueClasses = set(df_Train["Y"]) # Get classes from df_Train for context

    # Vote entropy per class #
    for classes in UniqueClasses:
        # Handle cases where `PredictedValues` might not contain a specific class
        # (e.g., if a tree never predicts it, or a batch doesn't have it)
        # Using `np.isin` for robustness
        class_votes = np.mean(PredictedValues == classes, axis=0) # proportion of committee members voting for this class
        
        # Ensure that LogVoteC handles cases where class_votes might be zero.
        # np.nan_to_num(np.log(class_votes)) will make log(0) into -inf, then into 0, effectively removing its contribution.
        log_class_votes = np.log(class_votes + np.finfo(float).eps) # Add a small epsilon to avoid log(0)
        
        # The original calculation for VoteEntropy[classes] was incorrect by taking mean over axis=0 and then summing over axis=1.
        # Vote Entropy is typically - sum_c (P_c * log P_c) over classes, for EACH data point.
        # P_c is the proportion of committee members that predict class c for a given data point.
        # PredictedValues is (num_candidate_obs) x (num_committee_members)

        # Let's recalculate Vote Entropy more accurately:
        # For each observation in X_Candidate_df (row `k` in PredictedValues)
        # count votes for each class `c`
        # calculate proportion P_c = (count of c / num_committee_members)
        # calculate - P_c * log(P_c)
        # sum over c.

    # Corrected Vote Entropy Calculation
    # Assuming PredictedValues is (N_samples, N_estimators)
    num_candidate_samples = PredictedValues.shape[0]
    num_estimators = PredictedValues.shape[1]
    
    VoteEntropyFinal = np.zeros(num_candidate_samples)

    for idx_sample in range(num_candidate_samples):
        sample_predictions = PredictedValues[idx_sample, :] # Predictions for this sample from all estimators
        
        # Calculate vote proportions for each class for this sample
        counts = pd.Series(sample_predictions).value_counts().reindex(UniqueClasses, fill_value=0)
        proportions = counts / num_estimators
        
        # Calculate entropy for this sample
        entropy_terms = -proportions * np.log(proportions + np.finfo(float).eps)
        VoteEntropyFinal[idx_sample] = np.sum(entropy_terms)
    
    # Measures #
    DiversityValues = df_Candidate["DiversityScores"]
    DensityValues = df_Candidate["DensityScores"]

    # Normalize #
    scaler = MinMaxScaler()
    # Reshape for scaler if they are Series, otherwise flatten
    DiversityValues_scaled = scaler.fit_transform(np.array(DiversityValues).reshape(-1, 1)).flatten()
    DensityValues_scaled = scaler.fit_transform(np.array(DensityValues).reshape(-1, 1)).flatten()
    VoteEntropyFinal_scaled = scaler.fit_transform(np.array(VoteEntropyFinal).reshape(-1, 1)).flatten()

    ### Uncertainty Metric ###
    df_Candidate["UncertaintyMetric"] = (1 - DiversityWeight - DensityWeight) * VoteEntropyFinal_scaled + \
                                        DiversityWeight * DiversityValues_scaled + \
                                        DensityWeight * DensityValues_scaled
    
    if df_Candidate.shape[0] >= BatchSize:
        IndexRecommendation = list(df_Candidate.sort_values(by="UncertaintyMetric", ascending=False).index[0:BatchSize])
    else:
        IndexRecommendation = list(df_Candidate.index)
        
    df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True) # Clean up temporary column

    # Output #
    Output["IndexRecommendation"] = IndexRecommendation

    return Output