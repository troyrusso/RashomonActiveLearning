import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd

# Define a function to extract error and time vectors
def ExtractErrorAndTime(files):
    error_vecs = []
    time_vecs = []
    for file in files:
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                error_vecs.append(data["ErrorVec"])
                time_vecs.append(data["ElapsedTime"])
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    return np.array(error_vecs), np.array(time_vecs)

# Command-line argument parser
parser = argparse.ArgumentParser(description="Aggregate simulation results.")
parser.add_argument("--DataType", type=str, required=True, help="Type of data.")
parser.add_argument("--ModelType", type=str, required=True, help="Prediction model type.")
parser.add_argument("--Categories", type=str, required=True, help="Comma-separated list of categories.")
args = parser.parse_args()

### Set Up ###
cwd = os.getcwd()
ResultsDirectory = os.path.join(cwd, "Results", args.DataType, args.ModelType)
OutputDirectory = os.path.join(ResultsDirectory, "ProcessedResults")
RawDirectory = os.path.join(ResultsDirectory, "Raw")
Categories = json.loads(args.Categories)

# ResultsDirectory = "/Users/simondn/Documents/RashomonActiveLearning/Results/COMPAS/TreeFarms"
# OutputDirectory = "/Users/simondn/Documents/RashomonActiveLearning/Results/COMPAS/TreeFarms/Processed"
# RawDirectory = "/Users/simondn/Documents/RashomonActiveLearning/Results/COMPAS/TreeFarms/Raw"
# Categories = ["PassiveLearning_MTTreeFarmsRashomonNum11.pkl",
# "RashomonQBC_MTTreeFarmsRashomonNum11.pkl",
# "RashomonQBC_MTTreeFarmsRashomonNum1010.pkl",
# "RashomonQBC_MTTreeFarmsRashomonNum100100.pkl"]

# Group files by category
category_files = {category: [] for category in Categories}
for filename in os.listdir(RawDirectory):
    if filename.endswith(".pkl"):
        for category in Categories:
            if filename.endswith(category):
                category_files[category].append(os.path.join(RawDirectory, filename))
                break

# Process files for each category
ErrorMatrices = {}
TimeMatrices = {}


for category, files in category_files.items():
    if not files:
        print(f"Warning: No files found for category {category}. Skipping.")
        continue
    print(f"Processing category: {category} with {len(files)} files")
    error_vecs, time_vecs = ExtractErrorAndTime(files)
    ErrorMatrices[category] = error_vecs  # Transpose
    TimeMatrices[category] = time_vecs    # Transpose


# Retain original category names as keys
ErrorMatrices = {category: ErrorMatrices[category] for category in category_files if category in ErrorMatrices}
TimeMatrices = {category: TimeMatrices[category] for category in category_files if category in TimeMatrices}
ErrorMatrices = {key.replace(".pkl", ""): value for key, value in ErrorMatrices.items()}
TimeMatrices = {key.replace(".pkl", ""): value for key, value in TimeMatrices.items()}

# Squeeze dimensions #
ErrorMatrices = {key: np.squeeze(matrix) for key, matrix in ErrorMatrices.items()}
TimeMatrices = {key: np.squeeze(matrix) for key, matrix in TimeMatrices.items()}

# Save results
os.makedirs(OutputDirectory, exist_ok=True)
for key, matrix in ErrorMatrices.items():
    np.savetxt(os.path.join(OutputDirectory, f"{key}_ErrorMatrix.csv"), matrix, delimiter=",")
for key, matrix in TimeMatrices.items():
    np.savetxt(os.path.join(OutputDirectory, f"{key}_TimeMatrix.csv"), matrix, delimiter=",")
