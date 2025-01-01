
### Packages ###
import os
import pandas as pd

### Function ###
def LoadAnalyzedData(data_type, base_directory, method, parameter):

    ResultsDirectory = os.path.join(base_directory, data_type, method)

    ### File Path ###

    if method == "RandomForestClassification":
        PathTemplates = {
            "Error_RF": f"ProcessedResults/ErrorVec/RF{parameter}_ErrorMatrix.csv",
            "Time_RF": f"ProcessedResults/ElapsedTime/RF{parameter}_TimeMatrix.csv",
            "SelectionHistory_RF": f"ProcessedResults/SelectionHistory/RF{parameter}_SelectionHistory.csv"
        }
    if method == "TreeFarms":
        PathTemplates = {
            "Error_UNREAL": f"ProcessedResults/ErrorVec/DPL{parameter}_ErrorMatrix.csv",
            "Error_DUREAL": f"ProcessedResults/ErrorVec/UNQ{parameter}_ErrorMatrix.csv",
            "Time_UNREAL": f"ProcessedResults/ElapsedTime/UNQ{parameter}_TimeMatrix.csv",
            "Time_DUREAL": f"ProcessedResults/ElapsedTime/DPL{parameter}_TimeMatrix.csv",
            "SelectionHistory_UNREAL": f"ProcessedResults/SelectionHistory/UNQ{parameter}_SelectionHistory.csv",
            "SelectionHistory_DUREAL": f"ProcessedResults/SelectionHistory/DPL{parameter}_SelectionHistory.csv",
            "TreeCounts_UNREAL": f"ProcessedResults/ElapsedTime/UNQ{parameter}_TimeMatrix.csv",
            "TreeCounts_DUREAL": f"ProcessedResults/ElapsedTime/DPL{parameter}_TimeMatrix.csv",
            "TreeCounts_ALL_UNREAL": f"ProcessedResults/TreeCount/UNQ{parameter}_AllTreeCount.csv",
            "TreeCounts_UNQ_UNREAL": f"ProcessedResults/TreeCount/UNQ{parameter}_UniqueTreeCount.csv",
        }

    #### Load Data Into Dictionary ###
    DataDictiomary = {}
    for key, RelativePath in PathTemplates.items():
        FullPath = os.path.join(ResultsDirectory, RelativePath)
        try:
            DataDictiomary[key] = pd.read_csv(FullPath)
        except FileNotFoundError:
            print(f"File not found: {FullPath}")
            DataDictiomary[key] = None

    return DataDictiomary
