# Summary: Runs active learning procedure by querying candidate observations from df_Candidate and adding them to the training set df_Train.
# Input:
#   df_Train: Training dataset.
#   df_Test: Test dataset.
#   df_Candidate: Candidate dataset.
#   SelectorType: Selector type. Examples can be GSx, GSy, or PassiveLearning.
#   SelectorArgs: Arguments needed for the selector model. For instance, GSx requires a distance metric.
#   ModelType: Predictive model. Examples can be LinearRegression or RandomForestRegresso.
#   ModelArgs: Arguments for the predictive model. For instance, the penalty for RidgeRegression.
# Output:
#   ErrorVec: A 1xM vector of errors with M being the number of observations in df_Candidate. 
#   SelectedObservationHistory: The index of the queried candidate observation at each iteration

### Import functions ###
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

    ### Algorithm ###
    for i in range(len(df_Candidate)):

        ### Prediction Model ###
        print("Iteration: " + str(i))
        ModelArgsFiltered = FilterArguments(ModelType, ModelArgs)
        Model = ModelType(**ModelArgsFiltered)
        if "Model" in SelectorArgs.keys(): SelectorArgs['Model'] = Model            # NOTE: THIS IS NOT DYNAMIC

        ### Current Error ###
        TestErrorVal = TestErrorFunction(Model, df_Test, ModelArgs["Type"])        # NOTE: Change to df_Test if there is a test set
        if(len(TestErrorVal) > 1):
            AllErrors = TestErrorVal                                                # Rashomon gives all errors of Rashomon
            CurrentError = float(np.min(AllErrors))                                 # Extract the best one
            SelectorArgs["AllErrors"] = AllErrors                                   # Use AllErrors in RashomonQBC Selector method
        else: 
            CurrentError = TestErrorVal                                             # One output for non-Rashomon
        ErrorVec.append(CurrentError)

        ### Sampling Procedure ###
        SelectorArgsFiltered = FilterArguments(SelectorType, SelectorArgs)
        QueryObservationIndex = SelectorType(**SelectorArgsFiltered)
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