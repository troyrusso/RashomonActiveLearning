# Summary: Runs one full iteration of the active learning process.
# Input:
#   DataFileInput: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder.
#   Seed: Seed for reproducability.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially "unseen" and later added to the training set.
#   SelectorType: Selector type. Examples can be GSx, GSy, or PassiveLearning.
#   ModelType: Predictive model. Examples can be LinearRegression or RandomForestRegresso.
#   DataArgs: Arguments N and K for when simulated data.
#   SelectorArgs: Arguments needed for the selector model. For instance, GSx requires a distance metric.
#   ModelArgs: Arguments for the predictive model. For instance, the penalty for RidgeRegression.
# Output: Simulation results that contain
#   ErrorVec: Vector of errors at each iteration of the learning process.
#   SelectionHistory: Vector of recommended index for query at each iteration of the learning process.
#   SimulationParameters: Parameters used in the simulation.
#   ElapsedTime: Time for the entire learning process.

### Import packages ###
import os
import time
import numpy as np
import math as math
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

### Function ###
def OneIterationFunction(DataFileInput,
                         Seed,
                         TestProportion,
                         CandidateProportion,
                         SelectorType, 
                         ModelType, 
                         DataArgs,
                         SelectorArgs,
                         ModelArgs):
    
    ### Run Time ###
    StartTime = time.time()

    ### Set Up ###
    random.seed(Seed)
    np.random.seed(Seed)
    ErrorVec = []
    SelectedObservationHistory = []

    ### Generate Data ###
    if(DataFileInput == "Simulate"):
        from utils.Main import DataGeneratingProcess                             ### NOTE: Why is this not imported from utils.Main import *
        df = DataGeneratingProcess(**DataArgs)
    else:
        df = LoadData(DataFileInput)

    ### Train Test Candidate Split
    from utils.Main import TrainTestCandidateSplit                           ### NOTE: Why is this not imported from utils.Main import *
    df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, TestProportion, CandidateProportion)

    ### Selector Arguments ###
    SelectorArgs["df_Train"] = df_Test                                     # NOTE: Change to df_Test if there is a test set
    SelectorArgs["df_Candidate"] = df_Candidate
    SelectorArgs["Model"] = ""
    # SelectorArgsFiltered = FilterArguments(SelectorType, SelectorArgs)

    ### Model Arguments ###
    ModelArgs['df_Train'] = df_Train
    # ModelArgsFiltered = FilterArguments(ModelType, ModelArgs)
    
    ### Learning Process ###
    from utils.Main import LearningProcedure                                 ### NOTE: Why is this not imported from utils.Main import *
    ErrorVec, SelectedObservationHistory = LearningProcedure(df_Train = df_Train, 
                                                                df_Test = df_Test, 
                                                                df_Candidate = df_Candidate, 
                                                                SelectorType = SelectorType, 
                                                                SelectorArgs = SelectorArgs,
                                                                ModelType = ModelType, 
                                                                ModelArgs = ModelArgs
                                                                )
    
    ### Return Simulation Parameters ###
    SimulationParameters = {"DataFileInput" : str(DataFileInput),
                            "Seed" : str(Seed),
                            "TestProportion" : str(TestProportion),
                            "CandidateProportion" : str(CandidateProportion),
                            "SelectorType" : str(SelectorType),
                            "ModelType" : str(ModelType),
                            "DataArgs" : str(DataArgs),
                            # "SelectorArgs" : str(SelectorArgs),
                            "ModelArgs" : str(FilterArguments(ModelType, ModelArgs).pop('df_Train', None))
                            }
    
    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    SimulationResults = {"ErrorVec" : pd.DataFrame(ErrorVec, columns =["Error"]),
                             "SelectionHistory" : pd.DataFrame(SelectedObservationHistory, columns = ["ObservationID"]),
                             "SimulationParameters" : SimulationParameters,
                             "ElapsedTime" : ElapsedTime}


    return SimulationResults