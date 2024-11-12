### Libraries ###
import pickle

def LoadData(filename):
    filepath = f"../Data/processed/{filename}"
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data
