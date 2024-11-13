### Libraries ###
import pickle
import pandas as pd

def LoadData(filename):
    filepath = f"../Data/processed/{filename}.pkl"
    with open(filepath, 'rb') as file:
        data = pickle.load(file).dropna()
    return data
