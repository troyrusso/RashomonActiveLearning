### Import Packages ###
import ast
import argparse
import json
import numpy as np
import math as math
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

### Get Directory ###
cwd = os.getcwd()
SaveDirectory = os.path.join(cwd, "Results")

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--JobName", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--Seed", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--Data", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--TestProportion", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--CandidateProportion", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--SelectorType", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--ModelType", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--DataArgs", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--SelectorArgs", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--ModelArgs", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--Output", type=str, default="-1", help="Simulation case number.")
args = parser.parse_args()

### Parameter Vector ###
SimulationConfigInput = {"DataFileInput": ParameterVector.iloc[i]["Data"],
                        "Seed": int(ParameterVector.iloc[i]["Seed"]),
                        "TestProportion": float(ParameterVector.iloc[i]["TestProportion"]),
                        "CandidateProportion": float(ParameterVector.iloc[i]["CandidateProportion"]),
                        "SelectorType": str(ParameterVector.iloc[i]["SelectorType"]), 
                        "ModelType": str(ParameterVector.iloc[i]["ModelType"]), 
                        "TopCModels": float(ParameterVector.iloc[i]["TopCModels"]), 
                        "UniqueErrorsInput": int(ParameterVector.iloc[i]["UniqueErrorsInput"]),
                        "n_estimators":int(ParameterVector.iloc[i]["n_estimators"]),
                        "regularization":float(ParameterVector.iloc[i]["regularization"]),
                        "rashomon_bound_adder":float(ParameterVector.iloc[i]["rashomon_bound_adder"]),
                        "Type":ParameterVector.iloc[i]["Type"]
                        }

### Run Code ###
SimulationResults = OneIterationFunction(SimulationConfigInput)

### Save Simulation Results ###
with open(os.path.join(SaveDirectory, str(args.Output)), 'wb') as f:
    pickle.dump(SimulationResults, f)