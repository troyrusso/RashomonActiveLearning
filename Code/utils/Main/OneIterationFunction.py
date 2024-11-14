### Import packages ###
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

    ### Set Up ###
    random.seed(Seed)
    np.random.seed(Seed)
    ErrorVec = []
    SelectedObservationHistory = []

    from utils.Main import DataGeneratingProcess

    ### Generate Data ###
    if(DataFileInput == "Simulate"):
        df = DataGeneratingProcess(**DataArgs)
    else:
        df = LoadData(DataFileInput)

    print("Good 2")

    ### Train Test Candidate Split
    df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, TestProportion, CandidateProportion)
    print("Good 3")

    ### Selector Arguments ###
    SelectorArgs["df_Train"] = df_Test
    SelectorArgs["df_Candidate"] = df_Candidate
    SelectorArgs["Model"] = ModelType
    SelectorArgsFiltered = FilterArguments(SelectorType, SelectorArgs)
    print("Good 4")

    ### Model Arguments ###
    ModelArgs['df_Train'] = df_Train
    ModelArgsFiltered = FilterArguments(ModelType, ModelArgs)
    print("Good 5")
    
    ### Learning Process ###
    ErrorVec, SelectedObservationHistory = LearningProcedure(df_Train = df_Train, 
                                                                df_Test = df_Test, 
                                                                df_Candidate = df_Candidate, 
                                                                SelectorType = SelectorType, 
                                                                SelectorArgs = SelectorArgsFiltered,
                                                                ModelType = ModelType, 
                                                                ModelArgs = ModelArgsFiltered
                                                                )
    print("Good 6")
    
    SimulationParameters = {"DataFileInput" : str(DataFileInput),
                            "Seed" : str(Seed),
                            "TestProportion" : str(TestProportion),
                            "CandidateProportion" : str(CandidateProportion),
                            "SelectorType" : str(SelectorType),
                            "ModelType" : str(ModelType),
                            "DataArgs" : str(DataArgs),
                            "SelectorArgs" : str(SelectorArgs),
                            "ModelArgs" : str(ModelArgs)}

    return pd.DataFrame(ErrorVec), pd.DataFrame(SelectedObservationHistory), SimulationParameters