### Import packages ###
import numpy as np
import time
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

    from utils.Main import DataGeneratingProcess                             ### NOTE: Why is this not imported from utils.Main import *?

    ### Generate Data ###
    if(DataFileInput == "Simulate"):
        df = DataGeneratingProcess(**DataArgs)
    else:
        df = LoadData(DataFileInput)

    ### Train Test Candidate Split
    from utils.Main import TrainTestCandidateSplit                           ### NOTE: Why is this not imported from utils.Main import *?
    df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, TestProportion, CandidateProportion)

    ### Selector Arguments ###
    SelectorArgs["df_Train"] = df_Test
    SelectorArgs["df_Candidate"] = df_Candidate
    SelectorArgs["Model"] = ModelType
    SelectorArgsFiltered = FilterArguments(SelectorType, SelectorArgs)

    ### Model Arguments ###
    ModelArgs['df_Train'] = df_Train
    ModelArgsFiltered = FilterArguments(ModelType, ModelArgs)
    
    ### Learning Process ###
    from utils.Main import LearningProcedure                                 ### NOTE: Why is this not imported from utils.Main import *?
    ErrorVec, SelectedObservationHistory = LearningProcedure(df_Train = df_Train, 
                                                                df_Test = df_Test, 
                                                                df_Candidate = df_Candidate, 
                                                                SelectorType = SelectorType, 
                                                                SelectorArgs = SelectorArgsFiltered,
                                                                ModelType = ModelType, 
                                                                ModelArgs = ModelArgsFiltered
                                                                )
    
    SimulationParameters = {"DataFileInput" : str(DataFileInput),
                            "Seed" : str(Seed),
                            "TestProportion" : str(TestProportion),
                            "CandidateProportion" : str(CandidateProportion),
                            "SelectorType" : str(SelectorType),
                            "ModelType" : str(ModelType),
                            "DataArgs" : str(DataArgs),
                            # "SelectorArgs" : str(SelectorArgs),
                            "ModelArgs" : str(ModelArgsFiltered.pop('df_Train', None))
                            }
    
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    SimulationResults = {"ErrorVec" : pd.DataFrame(ErrorVec, columns =["Error"]),
                             "SelectionHistory" : pd.DataFrame(SelectedObservationHistory, coluns = "ObservationID"),
                             "SimulationParameters" : SimulationParameters,
                             "ElapsedTime" : ElapsedTime}


    return SimulationResults