### Import functions ###
import sys
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *
import pandas as pd

### Function ###
def LearningProcedure(df_Train, 
                         df_Test, 
                         df_Candidate, 
                         SelectorType, 
                         SelectorArgs,
                         ModelType, 
                         ModelArgs):

    ### Set Up ###
    ErrorVec = []
    SelectedObservationHistory = []

    ### Algorithm
    for i in range(len(df_Candidate)):

        ### Prediction Model ###
        Model = ModelType(**ModelArgs)
        if "Model" in SelectorArgs.keys(): SelectorArgs['Model'] = Model            # NOTE: THIS IS NOT DYNAMIC

        ### Current Error ###
        CurrentError = TestErrorFunction(Model, df_Test)
        ErrorVec.append(CurrentError)

        ### Sampling Procedure ###
        QueryObservationIndex = SelectorType(**SelectorArgs)
        QueryObservation = df_Candidate.loc[[QueryObservationIndex]] # or should this be iloc
        SelectedObservationHistory.append(QueryObservationIndex)
        
        ### Update Train and Candidate Sets ###
        df_Train = pd.concat([df_Train, QueryObservation])
        df_Candidate = df_Candidate.drop(QueryObservationIndex)

        ### Update SelectorArgs and ModelArgs ###                                     # NOTE: THIS IS NOT DYNAMIC
        if "df_Train" in ModelArgs.keys(): ModelArgs['df_Train'] = df_Train
        if "df_Train" in SelectorArgs.keys(): SelectorArgs['df_Train'] = df_Train
        if "df_Candidate" in SelectorArgs.keys(): SelectorArgs['df_Candidate'] = df_Candidate      

    ### RETURN ###
    return ErrorVec, SelectedObservationHistory