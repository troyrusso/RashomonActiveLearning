# Summary:
# Input:
# Output:

### Libraries ###
import os
import pickle
import pandas as pd

def LoadData(DataFileInput):
    ### Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    directories = [cwd, ParentDirectory]  # Cluster first, then local

    ### Get Data ###
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".pkl")
            with open(filepath, 'rb') as file:
                data = pickle.load(file).dropna()
            return data
        except FileNotFoundError:
            continue
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")

    return data
