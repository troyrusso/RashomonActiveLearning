### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

### GSx ###
def GSxFunction(df_Train, df_Candidate, distance = "euclidean"):

    ### Calculate n*m distance from df_Candidate_n to df_Train_m
    d_nmX = cdist(df_Candidate.loc[:,df_Candidate.columns!= "Y"], df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)

    ### Find the nearest neighbor for each of the n observation in df_Train_n ###
    d_nX = d_nmX.min(axis=1)

    ### Return the index of the furthest nearest neighbor ###
    MaxRowNumber = np.argmax(d_nX)
    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

    return(IndexRecommendation)

### GSy ###
def GSyFunction(df_Train, df_Candidate, Model, distance = "euclidean"):  ### NOTE: or should df_Train be df_Test

    ### Calculate the prediction of df_Train_n ###
    Predictions = Model.predict(df_Candidate.loc[:, df_Candidate.columns != "Y"])
    print("Line 1 good")

    ### Calculate the difference between f(x_n) and y_m ###
    d_nmY = cdist(df_Train["Y"].values.reshape(-1,1), Predictions.reshape(-1,1), metric = distance)
    print("Line 2 good")

    ### Return the index of the furthest error ###
    d_nY = d_nmY.min(axis=1)
    print("Line 3 good")

    MaxRowNumber = np.argmax(d_nY)
    print("Line 4 good")

    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]
    print("Line 5 good")


    return(IndexRecommendation)
    
### iGS ###
def iGSFunction(df_Train, df_Candidate, Model, distance = "euclidean"):

    ### GSx ###
    d_nmX = cdist(df_Candidate.loc[:,df_Candidate.columns!= "Y"], df_Train.loc[:,df_Train.columns!= "Y"], metric = distance)
    d_nX = d_nmX.min(axis=1)

    ### GSy ###
    Predictions = Model.predict(df_Candidate.loc[:, df_Candidate.columns != "Y"])
    d_nmY = cdist(df_Train["Y"].values.reshape(-1,1), Predictions.reshape(-1,1), metric = distance)
    d_nY = d_nmY.min(axis=1)

    ### iGS ###
    d_nXY = d_nX*d_nY
    MaxRowNumber = np.argmax(d_nXY)
    IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]

    return(IndexRecommendation)