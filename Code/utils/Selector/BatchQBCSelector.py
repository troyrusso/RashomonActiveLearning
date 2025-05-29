# utils/Selector/BatchQBCSelector.py

# Summary: Query-by-committee selector that recommends observations from the candidate set to be queried,
#          incorporating diversity and density metrics.

### Libraries ###
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class BatchQBCSelector:
    def __init__(self, UniqueErrorsInput: int, DiversityWeight: float, DensityWeight: float, BatchSize: int, Seed: int = None, **kwargs):
        """
        Initializes the BatchQBCSelector.

        Args:
            UniqueErrorsInput (int): A binary input indicating whether to consider unique prediction patterns
                                     from the committee for vote entropy calculation (relevant for TreeFarms/ensemble models).
            DiversityWeight (float): The weight placed on the diversity of the data inputs for the selection mechanism.
            DensityWeight (float): The weight placed on the density of the data inputs for the selection mechanism.
            BatchSize (int): The number of observations to be queried in batch active learning.
            Seed (int, optional): Seed for reproducibility. (Not directly used by QBC logic, but for consistency).
            **kwargs: Catches any other arguments not used by this selector's init.
        """
        self.UniqueErrorsInput = UniqueErrorsInput
        self.DiversityWeight = DiversityWeight
        self.DensityWeight = DensityWeight
        self.BatchSize = BatchSize
        self.Seed = Seed 

        ### Ignore warning (taken care of) ###
        warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply", category=RuntimeWarning)

    def select(self, df_Candidate: pd.DataFrame, Model, df_Train: pd.DataFrame, auxiliary_columns: list = None) -> dict:
        """
        Selects observations using a Query-by-Committee (QBC) approach,
        combining vote entropy with diversity and density scores.

        Args:
            df_Candidate (pd.DataFrame): The candidate set from which to select observations.
            Model: The trained predictive model instance, expected to have a 'get_raw_ensemble_predictions' method.
            df_Train (pd.DataFrame): The current training set (used to determine unique classes).
            auxiliary_columns (list, optional): A list of column names that are not features.

        Returns:
            dict: Contains "IndexRecommendation" (List[int]) of the selected observations,
                  and potentially "AllTreeCount", "UniqueTreeCount" if available from the model.
        """
        # Prepare candidate features
        X_Candidate_df, _ = get_features_and_target(
            df=df_Candidate,
            target_column_name="Y", # Still needed to identify Y column if present
            auxiliary_columns=auxiliary_columns
        )

        ### Predicted Values (Committee Votes) ###
        if not hasattr(Model, 'get_raw_ensemble_predictions'):
            raise TypeError(f"Model type {type(Model).__name__} does not have 'get_raw_ensemble_predictions' "
                            f"and is not supported for BatchQBCSelector's current implementation.")

        PredictedValues_df = Model.get_raw_ensemble_predictions(X_Candidate_df)
        PredictedValues = PredictedValues_df.values 

        # Initialize Output dictionary for selector results
        Output = {}

        # Retrieve tree counts if the model provides them (like TreeFarmsPredictor)
        if hasattr(Model, 'get_tree_counts'):
            tree_counts = Model.get_tree_counts()
            Output["AllTreeCount"] = tree_counts.get("AllTreeCount", 0)
            Output["UniqueTreeCount"] = tree_counts.get("UniqueTreeCount", 0)
        else: 
            Output["AllTreeCount"] = PredictedValues_df.shape[1] 
            Output["UniqueTreeCount"] = PredictedValues_df.T.drop_duplicates().shape[0] 

        if self.UniqueErrorsInput and Output["UniqueTreeCount"] > 0: 
            PredictedValues_df_unique = PredictedValues_df.T.drop_duplicates().T
            PredictedValues = PredictedValues_df_unique.values
            num_estimators_for_entropy = PredictedValues_df_unique.shape[1]
        else:
            num_estimators_for_entropy = PredictedValues_df.shape[1]


        if PredictedValues is None or PredictedValues.size == 0:
            raise RuntimeError("PredictedValues could not be obtained or are empty for BatchQBCSelector.")

        ### Vote Entropy Calculation ###
        UniqueClasses = np.unique(df_Train["Y"])
        
        num_candidate_samples = PredictedValues.shape[0]
        VoteEntropyFinal = np.zeros(num_candidate_samples)

        for idx_sample in range(num_candidate_samples):
            sample_predictions = PredictedValues[idx_sample, :] 
            
            # Calculate vote proportions for each class for this sample
            counts = pd.Series(sample_predictions).value_counts().reindex(UniqueClasses, fill_value=0)
            proportions = counts / num_estimators_for_entropy 
            
            # Calculate entropy for this sample
            entropy_terms = -proportions * np.log(proportions + np.finfo(float).eps)
            VoteEntropyFinal[idx_sample] = np.sum(entropy_terms)
        
        # Measures #
        DiversityValues = df_Candidate["DiversityScores"]
        DensityValues = df_Candidate["DensityScores"]

        # Normalize #
        scaler = MinMaxScaler()
        DiversityValues_scaled = scaler.fit_transform(np.array(DiversityValues).reshape(-1, 1)).flatten()
        DensityValues_scaled = scaler.fit_transform(np.array(DensityValues).reshape(-1, 1)).flatten()
        VoteEntropyFinal_scaled = scaler.fit_transform(np.array(VoteEntropyFinal).reshape(-1, 1)).flatten()

        ### Uncertainty Metric ###
        df_Candidate["UncertaintyMetric"] = (1 - self.DiversityWeight - self.DensityWeight) * VoteEntropyFinal_scaled + \
                                            self.DiversityWeight * DiversityValues_scaled + \
                                            self.DensityWeight * DensityValues_scaled
        
        # Ensure BatchSize doesn't exceed available candidates
        batch_size_actual = min(self.BatchSize, df_Candidate.shape[0])

        if batch_size_actual > 0:
            IndexRecommendation = list(df_Candidate.sort_values(by="UncertaintyMetric", ascending=False).index[0:batch_size_actual])
        else:
            IndexRecommendation = []

        df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)

        # Output #
        Output["IndexRecommendation"] = IndexRecommendation

        return Output