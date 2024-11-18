# Summary:
# Input:
# Output:

### Libraries ###
import os
import pickle
import pandas as pd

def LoadData(filename):
    ### Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    # CurrentDirectory = ParentDirectory                              # NOTE: FOR LOCAL SIMULATIONS
    CurrentDirectory = cwd                                        # NOTE: FOR THE CLUSTER

    ### File Path ###
    CurrentDirectory = "/Users/simondn/Documents/RashomonActiveLearning"
    filepath = os.path.join(CurrentDirectory, "Data","processed", filename +".pkl")
    with open(filepath, 'rb') as file:
        data = pickle.load(file).dropna()
    return data
