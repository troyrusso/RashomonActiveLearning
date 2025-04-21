# Summary: Loads the three greedy sampling methods from Wu, Lin, and Huang (2018). 
#   GSx samples based on the covariate space, GSy based on the output space, and iGS on both.
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

### GSx ###
def GSxFunction(df_Train, df_Candidate, distance = "euclidean"):

    # Variables #
    columns_to_remove = ['Y', "DiversityScores", "DensityScores"]
    X_Candidate = df_Candidate[df_Candidate.columns.difference(columns_to_remove)]

    ### Calculate n*m distance from df_Candidate_n to df_Train_m
    d_nmX = cdist(X_Candidate, df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)

    ### Find the nearest neighbor for each of the n observation in df_Train_n ###
    d_nX = d_nmX.min(axis=1)

    ### Return the index of the furthest nearest neighbor ###
    MaxRowNumber = np.argmax(d_nX)
    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

    ### Output ###
    Output = {"IndexRecommendation": [float(IndexRecommendation)]}
    return(Output)

### GSy ###
def GSyFunction(df_Train, df_Candidate, Model, distance = "euclidean"): 

    ### Variables ###
    columns_to_remove = ['Y', "DiversityScores", "DensityScores"]
    X_Candidate = df_Candidate[df_Candidate.columns.difference(columns_to_remove)]

    ### Prediction ###
    Predictions = Model.predict(X_Candidate)

    ### Calculate the difference between f(x_n) and y_m ###
    d_nmY = cdist(Predictions.reshape(-1,1), df_Train["Y"].values.reshape(-1,1), metric = distance)

    ### Return the index of the furthest error ###
    d_nY = d_nmY.min(axis=1)
    MaxRowNumber = np.argmax(d_nY)
    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

    ### Output ###
    Output = {"IndexRecommendation": [float(IndexRecommendation)]}
    return(Output)
    
### iGS ###
def iGSFunction(df_Train, df_Candidate, Model, distance = "euclidean"):

    ### Variables ###
    columns_to_remove = ['Y', "DiversityScores", "DensityScores"]
    X_Candidate = df_Candidate[df_Candidate.columns.difference(columns_to_remove)]

    ### GSx ###
    d_nmX = cdist(X_Candidate, df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)
    d_nX = d_nmX.min(axis=1)

    ### GSy ###
    ## Prediction ##
    Predictions = Model.predict(X_Candidate)
    d_nmY = cdist(Predictions.reshape(-1,1), df_Train["Y"].values.reshape(-1,1), metric = distance)
    d_nY = d_nmY.min(axis=1)

    ### iGS ###
    d_nXY = d_nX*d_nY
    MaxRowNumber = np.argmax(d_nXY)
    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

    ### Output ###
    Output = {"IndexRecommendation": [float(IndexRecommendation)]}
    return(Output)
