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
#   rashomon_bound_adder: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss
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

### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

import json
import networkx as nx
import matplotlib.pyplot as plt


### Function ###
def OneIterationFunction(SimulationConfigInput):
    
    ### Set Up ###
    StartTime = time.time()
    random.seed(SimulationConfigInput["Seed"])
    np.random.seed(SimulationConfigInput["Seed"])

    ### Generate Data ###
    # if(DataFileInput == "Simulate"):
    #     from utils.Main import DataGeneratingProcess                             ### NOTE: Why is this not imported from utils.Main import *
    #     df = DataGeneratingProcess(**DataArgs)
    # else:
    df = LoadData(SimulationConfigInput["DataFileInput"])

    ### Train Test Candidate Split ###
    from utils.Main import TrainTestCandidateSplit                           ### NOTE: Why is this not imported from utils.Main import *
    df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, SimulationConfigInput["TestProportion"], SimulationConfigInput["CandidateProportion"])

    ### Update SimulationConfig Arguments ###
    SimulationConfigInput['df_Train'] = df_Train
    SimulationConfigInput["df_Test"] = df_Test                                     # NOTE: Change to df_Test if there is a test set
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
                            'rashomon_bound_adder': str(SimulationConfigInput["rashomon_bound_adder"]),
                            'Type': 'Classification'
                            }
    
    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    SimulationResults = {"ErrorVec" : pd.DataFrame(LearningProcedureOutput["ErrorVec"], columns =["Error"]),
                         "TreeCount": LearningProcedureOutput["TreeCount"],
                         "SelectionHistory" : pd.DataFrame(LearningProcedureOutput["SelectedObservationHistory"], 
                                                           columns = ["ObservationID"]),
                         "SimulationParameters" : SimulationParameters,
                         "ElapsedTime" : ElapsedTime}


    return SimulationResults