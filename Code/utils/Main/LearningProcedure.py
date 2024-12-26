# Summary: Runs active learning procedure by querying candidate observations from df_Candidate and adding them to the training set df_Train.
# Input: A dictionary SimulationConfigInputUpdated with the following keys and values:
#   DataFileInput: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder.
#   df_Train: The given train dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Test: The given test dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Candidate: The given candidate dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
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
# Output:
#   ErrorVec: A 1xM vector of errors with M being the number of observations in df_Candidate. 
#   SelectedObservationHistory: The index of the queried candidate observation at each iteration
#   RashomonCommitteeDict: A dictionary that contains two keys: {AllModelsInRashomonSet, UniqueModelsInRashomonSet} indicating
#                          the number of trees in the Rashomon set from TreeFarms and the number of unique classification patterns.

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
    RashomonCommitteeDict = {"AllModelsInRashomonSet": [], "UniqueModelsInRashomonSet": []}

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
        if('TREEFARMS' in str(type(Model))):                                                         # If Rashomon
            AllErrors = TestErrorVal                                                                 # All errors of Rashomon
            CurrentError = float(min(AllErrors))                                                     # Extract the best one
            RashomonCommitteeDict["AllModelsInRashomonSet"].append(Model.get_tree_count())           # Store number of trees
            RashomonCommitteeDict["UniqueModelsInRashomonSet"].append(len(set(AllErrors)))           # Store number of unique/duplicate trees
        else: 
            CurrentError = float(TestErrorVal[0])                                                    # One output for non-Rashomon
            AllErrors = [None]
        SimulationConfigInputUpdated["AllErrors"] = AllErrors                                        # Use AllErrors in RashomonQBC
        ErrorVec.append(CurrentError)

        print("CURRENT ERROR: ")
        print(CurrentError)
        print("---")

        ### Sampling Procedure ###
        SelectorType = globals().get(SimulationConfigInputUpdated["SelectorType"], None)
        SelectorArgsFiltered = FilterArguments(SelectorType, SimulationConfigInputUpdated)
        QueryObservationIndex = SelectorType(**SelectorArgsFiltered)
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[[QueryObservationIndex]] # or should this be iloc
        SelectedObservationHistory.append(QueryObservationIndex)
        
        ### Update Train and Candidate Sets ###
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex) 

    ### RETURN ###
    LearningProcedureOutput = {"ErrorVec": ErrorVec,
                               "RashomonCommitteeDict": RashomonCommitteeDict,
                               "SelectedObservationHistory": SelectedObservationHistory}
                              
    return LearningProcedureOutput