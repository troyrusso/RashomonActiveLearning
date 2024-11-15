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
    # HomeDirectory = os.path.abspath(os.path.join(cwd))

    ### File Path ###
    filepath = os.path.join(cwd, "Data","processed",filename +".pkl")
    with open(filepath, 'rb') as file:
        data = pickle.load(file).dropna()
    return data
