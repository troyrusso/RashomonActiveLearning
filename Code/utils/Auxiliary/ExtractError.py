### Import packages ###
import os
import re
import pickle
import argparse
import pandas as pd
print("Chunk 1 good!")

### Set Up  ###
cwd = os.getcwd()
ResultsDirectory = os.path.join(cwd, "Results")
SaveDirectory = os.path.join(ResultsDirectory, DataType, ProcessedResults)
print("Chunk 2 good!")

### Get Directory ###
cwd = os.getcwd()
SaveDirectory = os.path.join(cwd, "Results")
print("Chunk 3 good!")

### Parser ###
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--ResultsDirectory", type=str, default="-1", help="Results Directory.")
parser.add_argument("--DataType", type=str, default="-1", help="DataType.")
parser.add_argument("--SelectorType", type=str, default="-1", help="SelectorType.")
parser.add_argument("--ModelType", type=str, default="-1", help="ModelType.")
args = parser.parse_args()
print("Chunk 4 good!")

#### Construct Directory ###
Directory = os.path.join(args.ResultsDirectory, args.DataType, args.ModelType)
print("Chunk 5 good!")

### Initialize ###
AllSelectorMethodErrors = {}
print("Chunk 6 good!")

### Iterate over directory files ###
for FileName in os.listdir(Directory):
    if FileName.endswith(".pkl"):

        ### Extract ###
        match = re.match(r"Seed\d+_Data(\w+)_TP[\d.]+_CP[\d.]+_ST(\w+)_MT(\w+).pkl", FileName)
        
        ### Check Match ###
        if match:
            file_DataType = match.group(1)
            file_SelectorType = match.group(2)
            file_ModelType = match.group(3)
            
            ### Check Match ###
            if file_DataType == args.DataType and file_SelectorType == args.SelectorType and file_ModelType == args.ModelType:

                ### Load ###
                with open(os.path.join(Directory, FileName), "rb") as f:
                    SimulationResults = pickle.load(f)
                
                ### Extract the ErrorVec ###
                ErrorVector = SimulationResults["ErrorVec"]

                ### Append ###
                AllSelectorMethodErrors[file_SelectorType] = pd.concat([AllSelectorMethodErrors[file_SelectorType], ErrorVector], 
                                                                        ignore_index=True)
print("Chunk 7 good!")

#### Save to CSV ###
for file_SelectorType, df in AllSelectorMethodErrors.items():
    csv_file = os.path.join(SaveDirectory, f"{file_SelectorType}_ErrorVec.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved {file_SelectorType}_ErrorVec.csv")
print("Chunk 8 good!")
print("yaaaayyyy!!!!!")

