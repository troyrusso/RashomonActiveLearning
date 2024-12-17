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
def LearningProcedure(SimulationConfigInputUpdated):

    ### Set Up ###
    ErrorVec = []
    SelectedObservationHistory = []

    ### Algorithm ###
    for i in range(len(SimulationConfigInputUpdated["df_Candidate"])):

        ### Prediction Model ###
        print("Iteration: " + str(i))
        ModelType = globals().get(SimulationConfigInputUpdated["ModelType"], None)
        ModelArgsFiltered = FilterArguments(ModelType, SimulationConfigInputUpdated)
        Model = ModelType(**ModelArgsFiltered)
        SimulationConfigInputUpdated['Model'] = Model

        ### Current Error ###
        TestErrorVal = TestErrorFunction(Model, SimulationConfigInputUpdated["df_Test"], SimulationConfigInputUpdated["Type"])
        if(len(TestErrorVal) > 1):
            AllErrors = TestErrorVal                                                # Rashomon gives all errors of Rashomon
            CurrentError = float(np.min(AllErrors))                                 # Extract the best one
        else: 
            CurrentError = TestErrorVal                                             # One output for non-Rashomon
            AllErrors = [None]
        SimulationConfigInputUpdated["AllErrors"] = AllErrors                       # Use AllErrors in RashomonQBC
        ErrorVec.append(CurrentError)

        ### Sampling Procedure ###
        SelectorType = globals().get(SimulationConfigInputUpdated["SelectorType"], None)
        SelectorArgsFiltered = FilterArguments(SelectorType, SimulationConfigInputUpdated)
        QueryObservationIndex = SelectorType(**SelectorArgsFiltered)
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[[QueryObservationIndex]] # or should this be iloc
        SelectedObservationHistory.append(QueryObservationIndex)
        
        ### Update Train and Candidate Sets ###
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex)

        # ### Update SimulationConfigInputUpdated ###
        # SimulationConfigInputUpdated['df_Train'] = df_Train
        # SimulationConfigInputUpdated['df_Candidate'] = df_Candidate   

    ### RETURN ###
    return ErrorVec, SelectedObservationHistory