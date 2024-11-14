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
ParentDirectory = os.path.abspath(os.path.join(cwd, ".."))
SaveDirectory = os.path.join(ParentDirectory, "Results")


# ### Get Parameters ###
# ParameterVector = pd.read_csv(os.path.join(ParentDirectory, "Data", "raw", "ParameterVectorSimulations.csv"))

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

### Set Up ###
ErrorVecSimulation = []
HistoryVecSimulation = []

### Run Code ###
ErrorVec, HistoryVec, SimulationParameters = OneIterationFunction(DataFileInput = args.Data,
                                                Seed = int(args.Seed),
                                                TestProportion = float(args.TestProportion),
                                                CandidateProportion = float(args.CandidateProportion),
                                                SelectorType = globals().get(args.SelectorType, None), 
                                                ModelType = globals().get(args.ModelType, None), 
                                                DataArgs = json.loads(args.DataArgs),
                                                SelectorArgs = json.loads(args.SelectorArgs.replace("[","").replace("]","")),
                                                ModelArgs = json.loads(args.ModelArgs)
                                                )

### Save Error Vec ###
print(SaveDirectory)
ErrorVec.to_csv(os.path.join(SaveDirectory, str(args.Output)), index = False)
