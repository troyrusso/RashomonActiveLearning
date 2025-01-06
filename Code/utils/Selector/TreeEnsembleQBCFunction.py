# Summary: Query-by-committee function for either random forest or Rashomon's TreeFarms that 
#          recommends an observation from the candidate set to be queried.
# Input:
#   Model: The predictive model.
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
from scipy.spatial.distance import cdist

### Function ###
def TreeEnsembleQBCFunction(Model, df_Candidate, df_Train, UniqueErrorsInput):

    ### Ignore warning (taken care of) ###
    np.seterr(all = 'ignore') 
    warnings.filterwarnings("ignore", category=UserWarning)

    ### Predicted Values ###
    ## Rashomon Classification ##
    if 'TREEFARMS' in str(type(Model)):
        TreeCounts = Model.get_tree_count()

        # Duplicate #
        PredictionArray_Duplicate = pd.DataFrame(np.array([Model[i].predict(df_Candidate.loc[:, df_Candidate.columns != "Y"]) for i in range(TreeCounts)]))
        PredictionArray_Duplicate.columns = df_Candidate.index.astype(str)
        # EnsemblePrediction_Duplicate = np.mean(PredictionArray_Duplicate, axis =0)>=0.5
        EnsemblePrediction_Duplicate = pd.Series(stats.mode(PredictionArray_Duplicate)[0])
        EnsemblePrediction_Duplicate.index = df_Candidate["Y"].index
        AllTreeCount = PredictionArray_Duplicate.shape[0]

        # Unique #
        PredictionArray_Unique = pd.DataFrame(PredictionArray_Duplicate).drop_duplicates()
        # EnsemblePrediction_Unique = np.mean(PredictionArray_Unique, axis =0)>=0.5
        EnsemblePrediction_Unique = pd.Series(stats.mode(PredictionArray_Unique)[0])
        EnsemblePrediction_Unique.index = df_Candidate["Y"].index
        UniqueTreeCount = PredictionArray_Unique.shape[0]

        if UniqueErrorsInput:
            PredictedValues = PredictionArray_Unique
        else:
            PredictedValues = PredictionArray_Duplicate

        Output = {"AllTreeCount": AllTreeCount,
                  "UniqueTreeCount": UniqueTreeCount}

    elif 'RandomForestClassifier' in str(type(Model)):                                                          # RandomForest
        PredictedValues = [Model.estimators_[tree].predict(df_Candidate.loc[:, df_Candidate.columns != "Y"]) for tree in range(Model.n_estimators)] 
        PredictedValues = np.vstack(PredictedValues)
        Output = {}

    ### Vote Entropy ###
    VoteC = {}
    LogVoteC = {}
    VoteEntropy = {}
    UniqueClasses = set(df_Train["Y"])

    # Vote entropy per class #
    for classes in UniqueClasses:
        VoteC[classes] = np.mean(PredictedValues == classes, axis=0)
        LogVoteC[classes] = np.log(VoteC[classes])
        VoteEntropy[classes] =  - VoteC[classes] * LogVoteC[classes]
        VoteEntropy[classes] = np.nan_to_num(VoteEntropy[classes], nan=0)
        
    # Vote Entropy #
    VoteEntropyMatrix = np.stack(list(VoteEntropy.values()), axis=1)
    VoteEntropyFinal = np.sum(VoteEntropyMatrix, axis=1)

    ### Uncertainty Metric ###
    df_Candidate["UncertaintyMetric"] = VoteEntropyFinal

    IndexRecommendation = int(df_Candidate.sort_values(by = "UncertaintyMetric", ascending = False).index[0])
    df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)

    # Output #
    Output["IndexRecommendation"] = IndexRecommendation

    return Output