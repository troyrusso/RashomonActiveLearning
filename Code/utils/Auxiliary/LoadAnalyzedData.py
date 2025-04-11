
### Packages ###
import os
import pickle
import pandas as pd

### Function ###
def LoadAnalyzedData(data_type, base_directory, method, parameter):

    ResultsDirectory = os.path.join(base_directory, data_type, method)

    ### File Path ###
    if method == "RandomForestClassification":
        PathTemplates = {
            "Error": f"ProcessedResults/ErrorVec/{parameter}_ErrorMatrix.csv",
            "Time": f"ProcessedResults/ElapsedTime/{parameter}_TimeMatrix.csv",
            "SelectionHistory_RF": f"ProcessedResults/SelectionHistory/{parameter}_SelectionHistory.pkl"
        }
    if method == "TreeFarms":
        PathTemplates = {
            "Error_UNREAL": f"ProcessedResults/ErrorVec/UA{parameter}_ErrorMatrix.csv",
            "Error_DUREAL": f"ProcessedResults/ErrorVec/DA{parameter}_ErrorMatrix.csv",
            "Time_UNREAL": f"ProcessedResults/ElapsedTime/UA{parameter}_TimeMatrix.csv",
            "Time_DUREAL": f"ProcessedResults/ElapsedTime/DA{parameter}_TimeMatrix.csv",
            "SelectionHistory_UNREAL": f"ProcessedResults/SelectionHistory/UA{parameter}_SelectionHistory.pkl",
            "SelectionHistory_DUREAL": f"ProcessedResults/SelectionHistory/DA{parameter}_SelectionHistory.pkl",
            "TreeCounts_UNIQUE_UNREAL": f"ProcessedResults/TreeCount/UA{parameter}_UniqueTreeCount.csv",
            "TreeCounts_UNIQUE_DUREAL": f"ProcessedResults/TreeCount/DA{parameter}_UniqueTreeCount.csv",
            "TreeCounts_ALL_UNREAL": f"ProcessedResults/TreeCount/UA{parameter}_AllTreeCount.csv",
            "TreeCounts_ALL_DUREAL": f"ProcessedResults/TreeCount/DA{parameter}_AllTreeCount.csv",
        }

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
                with open(os.path.join(ResultsDirectory, RelativePath), 'rb') as file:
                    DataDictionary[key] = pickle.load(file)
            except FileNotFoundError:
                print(f"File not found: {FullPath}")
                DataDictionary[key] = None
    return DataDictionary
