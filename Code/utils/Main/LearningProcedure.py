# Summary: Runs active learning procedure by querying candidate observations from df_Candidate and adding them to the training set df_Train.
# Input: A dictionary SimulationConfigInputUpdated with the following keys and values:
#   DataFileInput: A string that indicates the name of the DataFrame in the Data folder.
#   df_Train: The given training dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Test: The given test dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Candidate: The given candidate dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   Seed: Seed for reproducibility.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially "unseen" and later added to the training set.
#   SelectorType: Selector mechanism in the active learning framework.
#   ModelType: Predictive model.
#   Model: The current model in the current active learning iteration.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFARMS.
#   n_estimators: The number of weak learners used for a random forest.
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThresholdType: A string {"Adder", "Multiplier"} indicating whether the Rashomon threshold is added or multiplied.
#   RashomonThreshold: A float indicating the Rashomon threshold in TreeFARMS.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
#   DiversityWeight: The weight placed on the diversity of the data inputs for the selection mechanism in batch active learning.
#   DensityWeight: The weight placed on the density of the data inputs for the selection mechanism in batch active learning.
#   BatchSize: The number of observations to be queried in batch active learning.
#   auxiliary_data_cols: Columns to exclude when training the model.
# Output:
#   ErrorVec: A 1xM vector of errors with M being the number of observations in df_Candidate. 
#   SelectedObservationHistory: The index of the queried candidate observation at each iteration
#   TreeCount: A dictionary that contains two keys: {AllTreeCount, UniqueTreeCount} indicating
#                          the number of trees in the Rashomon set from TreeFarms and the number of unique classification patterns.

### Import functions ###
import pandas as pd
import inspect 

### Functions ###
from utils.Selector import *
from utils.Prediction import *
from utils.Auxiliary import get_features_and_target

### Function ###
def LearningProcedure(SimulationConfigInputUpdated):

    ### Set Up ###
    i = 0
    ErrorVec = []
    SelectedObservationHistory = []
    TreeCount = {"AllTreeCount": [], "UniqueTreeCount": []}
    refit_frequency = SimulationConfigInputUpdated.get("RefitFrequency", 1)

    ### Initialize Mdodel ###
    ModelClass = globals().get(SimulationConfigInputUpdated["ModelType"], None)       # Initialize the model instance
    model_init_args = {k: v for k, v in SimulationConfigInputUpdated.items()          # Extract only relevant args for the ModelClass
                       if k in inspect.signature(ModelClass.__init__).parameters}
    predictor_model = ModelClass(**model_init_args)                                   # Create the model instance
    SimulationConfigInputUpdated['Model'] = predictor_model                           # Store this instance of the model

    ### Last batch of observations added (LFR) ###
    last_added_X_batch = None
    last_added_y_batch = None

    ### Algorithm ###
    while len(SimulationConfigInputUpdated["df_Candidate"]) > 0:

        ## Print iteration ##
        print(f"Iteration: {i}")

        ## Get features and target for the current training set ##
        X_train_df, y_train_series = get_features_and_target(
            df=SimulationConfigInputUpdated["df_Train"],
            target_column_name="Y",
            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', [])
        )

        ## REFIT VS. UPDATE ###
        if i == 0:                                                                  # Always fit on the first iteration
            predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)
        else:                                                                       # Subsequence iterations
            ### Check if model is LFR ###
            if isinstance(predictor_model, TreefarmsLFRPredictor) and (i % refit_frequency == 0):
                predictor_model.refit(
                    X_to_add=last_added_X_batch,
                    y_to_add=last_added_y_batch,
                    epsilon=SimulationConfigInputUpdated["RashomonThreshold"], 
                    verbose=True 
                )
            ### If not LFR, refit ###
            else:
                predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)

        ### Test Error ###
        TestErrorOutput = TestErrorFunction(InputModel=predictor_model,
                                            df_Test=SimulationConfigInputUpdated["df_Test"],
                                            Type=SimulationConfigInputUpdated["Type"],
                                            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', []))
        
        ## Store Errors ##
        CurrentError = TestErrorOutput["ErrorVal"] 
        ErrorVec.append(CurrentError)

        ## Sampling Procedure ##
        SelectorClass = globals().get(SimulationConfigInputUpdated["SelectorType"], None)   # Initialize the selector instance
        selector_init_args = {k: v for k, v in SimulationConfigInputUpdated.items()         # Extract only relevant args for the SelectorClass
                              if k in inspect.signature(SelectorClass.__init__).parameters}
        selector_model = SelectorClass(**selector_init_args)                                # Create the selector instance
        SelectorFuncOutput = selector_model.select(                                         # Call the 'select' method on the selector instance
            df_Candidate=SimulationConfigInputUpdated["df_Candidate"],
            df_Train=SimulationConfigInputUpdated["df_Train"], 
            Model=predictor_model, 
            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', [])
        )

        ## Query selected observation ##
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[QueryObservationIndex]
        SelectedObservationHistory.append(QueryObservationIndex)

        ## Store Newly Queried Observations for LFR ##
        last_added_X_batch, last_added_y_batch = get_features_and_target(
            df=QueryObservation,
            target_column_name="Y",
            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', []))

        
        ## Update Train and Candidate Sets ##
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation]).drop(columns=['DiversityScores', 'DensityScores'])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex)
        
        ## Store Number of (Unique) Trees ##
        if hasattr(predictor_model, 'get_tree_counts'): 
             tree_counts = predictor_model.get_tree_counts() 
             TreeCount["AllTreeCount"].append(tree_counts.get("AllTreeCount", 0)) 
             TreeCount["UniqueTreeCount"].append(tree_counts.get("UniqueTreeCount", 0))
        else: 
            TreeCount["AllTreeCount"].append(0) 
            TreeCount["UniqueTreeCount"].append(0)

        ## Increase iteration ##
        i+=1 

    ### Output ###
    LearningProcedureOutput = {"ErrorVec": ErrorVec,
                               "TreeCount": TreeCount,
                               "SelectedObservationHistory": SelectedObservationHistory}
                              
    return LearningProcedureOutput