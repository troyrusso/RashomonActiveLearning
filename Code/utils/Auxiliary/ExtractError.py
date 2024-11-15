### Import packages ###
import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd

# Manually define parser
parser = argparse.ArgumentParser(description="Your script description here")
parser.add_argument("--DataType", type=str, help="An example integer argument")
parser.add_argument("--SelectorType", type=str, help="An example string argument")
parser.add_argument("--ModelType", type=str, help="An example string argument")
args = parser.parse_args()

### Set Up  ###
cwd = os.getcwd()
ResultsDirectory = os.path.join(cwd, "Results", args.DataType, args.ModelType,)
SaveDirectory = os.path.join(ResultsDirectory, "ProcessedResults")
AllSelectorMethodErrors = {selector: [] for selector in args.SelectorType}

#### Construct Directory ###
Directory = os.path.join(ResultsDirectory, "Raw")

### Iterate over directory files ###
for SelectorTypeIteration in args.SelectorType:
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
                MatchCheck = (file_DataType == args.DataType and file_SelectorType == SelectorTypeIteration and file_ModelType == args.ModelType)
                if MatchCheck:
                    ### Load ###
                    with open(os.path.join(Directory, FileName), "rb") as f:
                        SimulationResults = pickle.load(f)
                    
                    ### Extract the ErrorVec ###
                    ErrorVector = SimulationResults["ErrorVec"]

                    ### Append ### 
                    AllSelectorMethodErrors[SelectorTypeIteration].append(ErrorVector)
    AllSelectorMethodErrors[SelectorTypeIteration] = pd.DataFrame(np.array(AllSelectorMethodErrors[SelectorTypeIteration]).squeeze())

#### Save to CSV ###
for file_SelectorType, df in AllSelectorMethodErrors.items():
    csv_file = os.path.join(SaveDirectory, f"{file_SelectorType}_ErrorVec.csv")
    df.to_csv(csv_file, index=False, header= False)

