### Import libraries ###
import os
import pickle
import argparse
import numpy as np
import pandas as pd

### Extract Error and Time Function ###
def ExtractInformation(files):

    ### Set Up ###

    ModelIndicesVecStorage = []
    ThresholdValuesStorage = []
    Epsilon_F1ScoreStorage = []
    Epsilon_ClassAccuracyStorage = []
    
    for file in files:
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                ModelIndicesVecStorage.append(data["ModelIndicesVec"])
                ThresholdValuesStorage.append(data["ThresholdValues"])
                Epsilon_F1ScoreStorage.append(data["Epsilon_F1Score"])
                Epsilon_ClassAccuracyStorage.append(data["Epsilon_ClassAccuracy"])
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    return ModelIndicesVecStorage, ThresholdValuesStorage, Epsilon_F1ScoreStorage, Epsilon_ClassAccuracyStorage

### Parser ###
parser = argparse.ArgumentParser(description="Aggregate simulation results.")
parser.add_argument("--DataType", type=str, required=True, help="Type of data.")
args = parser.parse_args()

### Set Up ###
cwd = os.getcwd()
ResultsDirectory = os.path.join(cwd, "Results")
OutputDirectory = os.path.join(ResultsDirectory, "OptimalThreshold", "OptimalValues")
RawDirectory = os.path.join(ResultsDirectory, args.DataType, "Raw")

### Extract File Names ###
CategoryFileNames = []
for filename in os.listdir(RawDirectory):
    CategoryFileNames.append(os.path.join(RawDirectory, filename))

### Extract Data ###
ModelIndicesVecStorage, ThresholdValuesStorage, Epsilon_F1ScoreStorage, Epsilon_ClassAccuracyStorage = ExtractInformation(CategoryFileNames)
# ModelIndicesVecStorage = pd.DataFrame(ModelIndicesVecStorage.squeeze())
ThresholdValuesStorage = pd.DataFrame(ThresholdValuesStorage.squeeze())
Epsilon_F1ScoreStorage = pd.DataFrame(Epsilon_F1ScoreStorage.squeeze())
Epsilon_ClassAccuracyStorage = pd.DataFrame(Epsilon_ClassAccuracyStorage.squeeze())

### Save ###
ModelIndicesVecStorage.to_csv(os.path.join(OutputDirectory, "ErrorVec", f"{args.DataType}_ModelIndicesVecStorage.csv"), index=False)
ThresholdValuesStorage.to_csv(os.path.join(OutputDirectory, "ElapsedTime", f"{args.DataType}_ThresholdValuesStorage.csv"), index=False)
Epsilon_F1ScoreStorage.to_csv(os.path.join(OutputDirectory, "TreeCount", f"{args.DataType}_Epsilon_F1ScoreStorage.csv"), index=False)
Epsilon_ClassAccuracyStorage.to_csv(os.path.join(OutputDirectory, "TreeCount", f"{args.DataType}_Epsilon_ClassAccuracyStorage.csv"), index=False)
print(f"Saved {Category} files!")
