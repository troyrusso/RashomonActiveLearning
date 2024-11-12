### Import Packages ###
import ast
import argparse
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


### Get Parameters ###
ParameterVector = pd.read_csv(os.path.join(ParentDirectory, "Data", "raw", "ParameterVectorSimulations.csv"))

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
ErrorVec, HistoryVec = OneIterationFunction(DataFileInput = ParameterVector.iloc[args.Data],
                                                Seed = ParameterVector.iloc[args.Seed],
                                                TestProportion = ParameterVector.iloc[args.TestProportion],
                                                CandidateProportion = ParameterVector.iloc[args.CandidateProportion],
                                                SelectorType = globals().get(ParameterVector.iloc[args.SelectorType], None), 
                                                ModelType = globals().get(ParameterVector.iloc[args.ModelType], None), 
                                                DataArgs = ast.literal_eval(ParameterVector.iloc[args.DataArgs]),
                                                SelectorArgs = ast.literal_eval(ParameterVector.iloc[args.SelectorArgs].replace("[","").replace("]","")),
                                                ModelArgs = ast.literal_eval(ParameterVector.iloc[args.ModelArgs])
                                                )

### Save Error Vec ###
ErrorVec.to_csv(os.path.join(SaveDirectory, str(args.ModelArgs.Output)))
