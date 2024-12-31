
### Packages ###
import os
import pandas as pd

### Function ###
def LoadAnalyzedData(data_type, base_directory, parameter):

    ResultsDirectory = os.path.join(base_directory, data_type)

    ### File Path ###
    PathTemplates = {
        "Error_UNREAL": f"TreeFarms/ProcessedResults/ErrorVec/DPL{parameter}_ErrorMatrix.csv",
        "Error_DUREAL": f"TreeFarms/ProcessedResults/ErrorVec/UNQ{parameter}_ErrorMatrix.csv",
        "Time_UNREAL": f"TreeFarms/ProcessedResults/ElapsedTime/UNQ{parameter}_TimeMatrix.csv",
        "Time_DUREAL": f"TreeFarms/ProcessedResults/ElapsedTime/DPL{parameter}_TimeMatrix.csv",
        "SelectionHistory_UNREAL": f"TreeFarms/ProcessedResults/SelectionHistory/UNQ{parameter}_SelectionHistory.csv",
        "SelectionHistory_DUREAL": f"TreeFarms/ProcessedResults/SelectionHistory/DPL{parameter}_SelectionHistory.csv",
        "TreeCounts_UNREAL": f"TreeFarms/ProcessedResults/ElapsedTime/UNQ{parameter}_TimeMatrix.csv",
        "TreeCounts_DUREAL": f"TreeFarms/ProcessedResults/ElapsedTime/DPL{parameter}_TimeMatrix.csv",
        "TreeCounts_ALL_UNREAL": f"TreeFarms/ProcessedResults/TreeCount/UNQ{parameter}_AllTreeCount.csv",
        "TreeCounts_UNQ_UNREAL": f"TreeFarms/ProcessedResults/TreeCount/UNQ{parameter}_UniqueTreeCount.csv",
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
