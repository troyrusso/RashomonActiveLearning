# Summary: Runs active learning procedure by querying candidate observations from df_Candidate and adding them to the training set df_Train.
# Input: A dictionary SimulationConfigInputUpdated with the following keys and values:
#   DataFileInput: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder.
#   df_Train: The given train dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Test: The given test dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Candidate: The given candidate dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   Seed: Seed for reproducibility.
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
#   TreeCount: A dictionary that contains two keys: {AllTreeCount, UniqueTreeCount} indicating
#                          the number of trees in the Rashomon set from TreeFarms and the number of unique classification patterns.

### Import functions ###
import pandas as pd
import inspect 

# Import all modules from utils, ensuring new classes are available via globals()
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

from utils.Prediction.BayesianNeuralNetworkPredictor import BayesianNeuralNetworkPredictor
from utils.Prediction.RandomForestClassifierPredictor import RandomForestClassifierPredictor
from utils.Prediction.RandomForestRegressorPredictor import RandomForestRegressorPredictor
from utils.Prediction.LinearRegressionPredictor import LinearRegressionPredictor
from utils.Prediction.RidgeRegressionPredictor import RidgeRegressionPredictor
from utils.Prediction.TreeFarmsPredictor import TreeFarmsPredictor
# from utils.Prediction.TreefarmsLFRPredictor import TreefarmsLFRPredictor


### Function ###
def LearningProcedure(SimulationConfigInputUpdated):

    ### Set Up ###
    i = 0
    ErrorVec = []
    SelectedObservationHistory = []
    TreeCount = {"AllTreeCount": [], "UniqueTreeCount": []}

    # Initialize the model instance *once* before the loop
    ModelClass = globals().get(SimulationConfigInputUpdated["ModelType"], None)
    
    if ModelClass is None:
        raise ValueError(f"ModelType '{SimulationConfigInputUpdated['ModelType']}' not found. "
                         f"Please ensure it's correctly named and imported in utils/Prediction.")

    # Extract only relevant args for the ModelClass __init__
    model_init_args = {k: v for k, v in SimulationConfigInputUpdated.items() 
                       if k in inspect.signature(ModelClass.__init__).parameters}
    
    # Create the model instance
    predictor_model = ModelClass(**model_init_args) 

    # Pass this instance around instead of the class itself
    SimulationConfigInputUpdated['Model'] = predictor_model 

    ### Algorithm ###
    while len(SimulationConfigInputUpdated["df_Candidate"]) > 0:
        print(f"Iteration: {i}")

        # Get features and target for the current training set
        X_train_df, y_train_series = get_features_and_target(
            df=SimulationConfigInputUpdated["df_Train"],
            target_column_name="Y",
            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', [])
        )
        
        # Train Prediction Model: Always call fit for now.
        predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)

        ### Test Error ###
        TestErrorOutput = TestErrorFunction(InputModel=predictor_model, # Pass the instance
                                            df_Test=SimulationConfigInputUpdated["df_Test"],
                                            Type=SimulationConfigInputUpdated["Type"],
                                            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', []))
        
        # All models now return 'ErrorVal' consistently from TestErrorFunction
        CurrentError = TestErrorOutput["ErrorVal"] 
        ErrorVec.append(CurrentError)

        ### Sampling Procedure ###
        SelectorType = globals().get(SimulationConfigInputUpdated["SelectorType"], None)
        
        if SelectorType is None:
            raise ValueError(f"SelectorType '{SimulationConfigInputUpdated['SelectorType']}' not found. "
                             f"Please ensure it's correctly named and imported in utils/Selector.")

        # Filter arguments for the selector function/class (selectors are still functions for now)
        temp_selector_args = FilterArguments(SelectorType, SimulationConfigInputUpdated)
        temp_selector_args['auxiliary_columns'] = SimulationConfigInputUpdated.get('auxiliary_data_cols', [])
        
        # If the selector function expects a 'Model' argument, pass the predictor_model instance
        if 'Model' in inspect.signature(SelectorType).parameters:
            temp_selector_args['Model'] = predictor_model
        
        SelectorFuncOutput = SelectorType(**temp_selector_args)
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[QueryObservationIndex]
        SelectedObservationHistory.append(QueryObservationIndex)
        
        ### Update Train and Candidate Sets ###
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation]).drop(columns=['DiversityScores', 'DensityScores'])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex)
        
        ### Store Number of (Unique) Trees ###
        # Check for the correct method name 'get_tree_counts'
        if hasattr(predictor_model, 'get_tree_counts'): 
             tree_counts = predictor_model.get_tree_counts() 
             TreeCount["AllTreeCount"].append(tree_counts.get("AllTreeCount", 0)) 
             TreeCount["UniqueTreeCount"].append(tree_counts.get("UniqueTreeCount", 0))
        else: 
            TreeCount["AllTreeCount"].append(0) 
            TreeCount["UniqueTreeCount"].append(0)

        # Increase iteration #
        i+=1 

    ### RETURN ###
    LearningProcedureOutput = {"ErrorVec": ErrorVec,
                               "TreeCount": TreeCount,
                               "SelectedObservationHistory": SelectedObservationHistory}
                              
    return LearningProcedureOutput