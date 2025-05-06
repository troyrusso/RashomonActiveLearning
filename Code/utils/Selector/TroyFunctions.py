# Summary: Loads the our adaptation of the greedy sampling method iGS from Wu, Lin, and Huang (2018).
# Input:
#   df_Train: The training set.
#   df_Candidate: The candidate set.
#   Model: The predictive model.
#   distance: The distance metric.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.


### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler

### iGS ###
def WiGSFunction(df_Train, df_Candidate, Model, w, distance = "euclidean"):

    ### Variables ###
    scaler = StandardScaler()
    columns_to_remove = ['Y', "DiversityScores", "DensityScores"]
    X_Candidate = df_Candidate[df_Candidate.columns.difference(columns_to_remove)]

    ### GSx ###
    d_nmX = cdist(X_Candidate, df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)
    d_nX = d_nmX.min(axis=1)

    ### GSy ###
    Predictions = Model.predict(X_Candidate)
    d_nmY = cdist(Predictions.reshape(-1,1), df_Train["Y"].values.reshape(-1,1), metric = distance)
    d_nY = d_nmY.min(axis=1)

    ### Normalize ###
    d_nX = scaler.fit_transform(d_nX.reshape(-1, 1))
    d_nY = scaler.fit_transform(d_nY.reshape(-1, 1))
    
    ### iGS ###
    d_nXY_weighted = (1 - w) * d_nX + w * d_nY
    MaxRowNumber = np.argmax(d_nXY_weighted)
    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

    ### Output ###
    Output = {"IndexRecommendation": [float(IndexRecommendation)],
              "weight": w}
    return(Output)