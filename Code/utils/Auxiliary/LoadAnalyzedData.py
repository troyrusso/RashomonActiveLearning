
### Packages ###
import os
import pickle
import pandas as pd

### Function ###

### Function ###
def LoadAnalyzedData(data_type, base_directory, method, parameter):

    ResultsDirectory = os.path.join(base_directory, data_type, method)

    ### File Path ###
    if method == "RandomForestClassification":
        PathTemplates = {
            "Error": f"ProcessedResults/ErrorVec/{parameter}_ErrorMatrix.csv",
            "Time": f"ProcessedResults/ElapsedTime/{parameter}_TimeMatrix.csv",
            "SelectionHistory": f"ProcessedResults/SelectionHistory/{parameter}_SelectionHistory.pkl" 
        }
    elif method == "BayesianNeuralNetwork":
        PathTemplates = {
            "Error": f"ProcessedResults/ErrorVec/{parameter}_ErrorMatrix.csv",
            "Time": f"ProcessedResults/ElapsedTime/{parameter}_TimeMatrix.csv",
            "SelectionHistory": f"ProcessedResults/SelectionHistory/{parameter}_SelectionHistory.pkl"
            }
    elif method == "TreeFarms": 
        PathTemplates = {
            "Error_UNREAL": f"ProcessedResults/ErrorVec/U{parameter}_ErrorMatrix.csv", 
            "Error_DUREAL": f"ProcessedResults/ErrorVec/D{parameter}_ErrorMatrix.csv", 
            "Time_UNREAL": f"ProcessedResults/ElapsedTime/U{parameter}_TimeMatrix.csv",
            "Time_DUREAL": f"ProcessedResults/ElapsedTime/D{parameter}_TimeMatrix.csv",
            "SelectionHistory_UNREAL": f"ProcessedResults/SelectionHistory/U{parameter}_SelectionHistory.pkl",
            "SelectionHistory_DUREAL": f"ProcessedResults/SelectionHistory/D{parameter}_SelectionHistory.pkl",
            "TreeCounts_UNIQUE_UNREAL": f"ProcessedResults/TreeCount/U{parameter}_UniqueTreeCount.csv",
            "TreeCounts_UNIQUE_DUREAL": f"ProcessedResults/TreeCount/D{parameter}_UniqueTreeCount.csv",
            "TreeCounts_ALL_UNREAL": f"ProcessedResults/TreeCount/U{parameter}_AllTreeCount.csv",
            "TreeCounts_ALL_DUREAL": f"ProcessedResults/TreeCount/D{parameter}_AllTreeCount.csv",
        }
    else: 
        print(f"Warning: Unknown method '{method}'. No specific path templates defined.")
        return {}


    #### Load Data Into Dictionary ###
    DataDictionary = {}
    for key, RelativePath in PathTemplates.items():

        # Path #
        FullPath = os.path.join(ResultsDirectory, RelativePath)

        # CSV Files #
        if RelativePath.endswith(".csv"):
            try:
                DataDictionary[key] = pd.read_csv(FullPath)
            except FileNotFoundError:
                print(f"File not found: {FullPath}")
                DataDictionary[key] = None

        # PKL Files #
        if RelativePath.endswith(".pkl"):
            try:
                with open(FullPath, 'rb') as file: 
                    DataDictionary[key] = pickle.load(file)
            except FileNotFoundError:
                print(f"File not found: {FullPath}")
                DataDictionary[key] = None
    return DataDictionary