### Libraries ###
import numpy as np
import math as math
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

### Function ###
def RunLearningProcedure(df_Train, df_Test, df_Candidate, 
                         SelectorType, selector_args,
                         ModelType, model_args):

    ### Set Up ###
    ErrorVec = []
    SelectedObservationHistory = []

    ### Algorithm
    for i in range(0, len(df_Candidate)):

        ### Prediction Model ###
        Model = ModelType(**model_args)
        if "Model" in selector_args.keys(): selector_args['Model'] = Model            # NOTE: THIS IS NOT DYNAMIC

        CurrentError = TestErrorFunction(Model, df_Test)
        ErrorVec.append(CurrentError)

        ### Sampling Procedure ###
        QueryObservationIndex = SelectorType(**selector_args)
        QueryObservation = df_Candidate.loc[[QueryObservationIndex]] # or should this be iloc
        SelectedObservationHistory.append(QueryObservationIndex)

        # print("Iteration: ", i, "| QueryIndex: ", QueryObservationIndex, "| Inclusion: ", QueryObservationIndex in df_Candidate.index)
        # print(df_Train)
        # print(df_Candidate)
        # print("---")
        
        ### Update Train and Candidate Sets ###
        df_Train = pd.concat([df_Train, QueryObservation])
        df_Candidate = df_Candidate.drop(QueryObservationIndex)

        ### Update selector_args and model_args ###                                     # NOTE: THIS IS NOT DYNAMIC
        if "df_Train" in model_args.keys(): model_args['df_Train'] = df_Train
        if "df_Train" in selector_args.keys(): selector_args['df_Train'] = df_Train
        if "df_Candidate" in selector_args.keys(): selector_args['df_Candidate'] = df_Candidate            

    return ErrorVec, SelectedObservationHistory