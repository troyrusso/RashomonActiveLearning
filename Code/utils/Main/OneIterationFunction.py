# Summary: Runs one full iteration of the active learning process.
# Input: A dictionary SimulationConfigInput with the following keys and values:
#   DataFileInput: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder.
#   Seed: Seed for reproducability.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially "unseen" and later added to the training set.
#   SelectorType: Selector type. Examples can be GSx, GSy, or PassiveLearning.
#   ModelType: Predictive model. Examples can be LinearRegression or RandomForestRegresso.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFarms.
#   n_estimators: The number of trees for a random forest.
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThreshold: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output: A dictionary SimulationResults with the following keys and values:
#   ErrorVec: Vector of errors at each iteration of the learning process.
#   TreeCount: A dictionary that contains two keys: {AllModelsInRashomonSet, UniqueModelsInRashomonSet} indicating
#                          the number of trees in the Rashomon set from TreeFarms and the number of unique classification patterns.
#   SelectionHistory: Vector of recommended index for query at each iteration of the learning process.
#   SimulationParameters: Parameters used in the simulation.
#   ElapsedTime: Time for the entire learning process.

### Import packages ###
import time
import numpy as np
import math as math
import pandas as pd
import random as random
from sklearn.cluster import AgglomerativeClustering


### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

# import json
# import networkx as nx
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

### Function ###
def OneIterationFunction(SimulationConfigInput):
    
    ### Set Up ###
    StartTime = time.time()
    random.seed(SimulationConfigInput["Seed"])
    np.random.seed(SimulationConfigInput["Seed"])

    ### Load Data ###
    df = LoadData(SimulationConfigInput["DataFileInput"])

    ### Train Test Candidate Split ###
    from utils.Main import TrainTestCandidateSplit                           ### NOTE: Why is this not imported from utils.Main import *
    df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, SimulationConfigInput["TestProportion"], SimulationConfigInput["CandidateProportion"])

    ### Batch Active Learning Metrics ###
    # Set Up #
    X_Candidate = df_Candidate.loc[:, df_Candidate.columns!= "Y"]
    X_Train = df_Train.loc[:,df_Train.columns!= "Y"]

    # Clustering #
    cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
    ClusterLabels = cluster.fit_predict(X_Candidate)
    df_Candidate["ClusterLabels"] = ClusterLabels

    # Diversity Metric #
    d_nmX = cdist(X_Candidate, X_Train, metric = "euclidean")
    d_nX = d_nmX.min(axis=1)
    df_Candidate["d_nX"] = d_nX

    ### Update SimulationConfig Arguments ###
    SimulationConfigInput['df_Train'] = df_Train
    SimulationConfigInput["df_Test"] = df_Test
    SimulationConfigInput["df_Candidate"] = df_Candidate
    
    ### Learning Process ###
    from utils.Main import LearningProcedure                                 ### NOTE: Why is this not imported from utils.Main import *
    LearningProcedureOutput = LearningProcedure(SimulationConfigInputUpdated = SimulationConfigInput)
    
    ### Return Simulation Parameters ###
    SimulationParameters = {"DataFileInput" : str(SimulationConfigInput["DataFileInput"]),
                            "Seed" : str(SimulationConfigInput["Seed"]),
                            "TestProportion" : str(SimulationConfigInput["TestProportion"]),
                            "CandidateProportion" : str(SimulationConfigInput["CandidateProportion"]),
                            "SelectorType" :  str(SimulationConfigInput["SelectorType"]),
                            "ModelType" :  str(SimulationConfigInput["ModelType"]),
                            'UniqueErrorsInput': str(SimulationConfigInput["UniqueErrorsInput"]),
                            'n_estimators': str(SimulationConfigInput["n_estimators"]),
                            'regularization': str(SimulationConfigInput["regularization"]),
                            'RashomonThresholdType': str(SimulationConfigInput["RashomonThresholdType"]),
                            'RashomonThreshold': str(SimulationConfigInput["RashomonThreshold"]),
                            'Type': 'Classification',
                            'DiversityWeight': str(SimulationConfigInput["DiversityWeight"]),
                            'BatchSize': str(SimulationConfigInput["BatchSize"])
                            }
    
    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    SimulationResults = {"ErrorVec" : pd.DataFrame(LearningProcedureOutput["ErrorVec"], columns =["Error"]),
                         "TreeCount": LearningProcedureOutput["TreeCount"],
                         "SelectionHistory" : LearningProcedureOutput["SelectedObservationHistory"],
                         "SimulationParameters" : SimulationParameters,
                         "ElapsedTime" : ElapsedTime}


    return SimulationResults