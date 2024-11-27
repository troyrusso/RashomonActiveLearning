# Summary: 
# Input:
#   df_Train: The training set.
#   df_Candidate: The candidate set.
#   Model: The predictive model.
#   distance: The distance metric.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

# NOTE: Incorporate covariate GSx in selection criteria? Good for tie breakers.

### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

### Function ###
def RashomonQBCFunction(Model, df_Candidate, df_Train, TopCModels, AllErrors):

    # ### GSx ### Incorporate? Good for tie breakers.
    # d_nmX = cdist(df_Candidate.loc[:,df_Candidate.columns!= "Y"], df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)
    # d_nX = d_nmX.min(axis=1)

    ### Extract Errors ###
    LowestErrorIndices = np.argsort(AllErrors)[:TopCModels]

    ### Prediction ###
    PredictedValues = [Model[i].predict(df_Candidate) for i in LowestErrorIndices]
    RashomonMean = np.array(PredictedValues).mean(axis =0)

    ### Uncertainty Metric ###
    df_Candidate["UncertaintyMetric"] = abs(RashomonMean - 0.5)
    IndexRecommendation = int(df_Candidate.sort_values(by = "UncertaintyMetric", ascending = True).index[0])

    return(IndexRecommendation)

