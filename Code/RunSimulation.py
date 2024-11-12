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

### Get Parameters ###
ParameterVector = pd.read_csv("/Users/simondn/Documents/RashomonActiveLearning/Data/raw/ParameterVectorSimulations.csv")


############################################################################################################################################

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")

# Define command-line arguments
parser.add_argument("--JobName", type=str, default="Bad", help="Job Name")
parser.add_argument("--Seed", type=str, default="Bad", help="Seed")
parser.add_argument("--Data", type=int, default=69, help="Data")
parser.add_argument("--TestProportion", type=str, default="Bad", help="Test proportion")
parser.add_argument("--CandidateProportion", type=str, default="Bad", help="Candidate proportion")
parser.add_argument("--SelectorType", type=int, default=69, help="Selector type")
parser.add_argument("--ModelType", type=int, default=69, help="Model type")
parser.add_argument("--DataArgs", type=float, default=69, help="Correlation between covariate 1 and 2")
parser.add_argument("--SelectorArgs", type=int, default=69, help="Discretizes data into NBins")
parser.add_argument("--ModelArgs", type=float, default=69, help="Test set proportion")
parser.add_argument("--Output", type=float, default=69, help="Test set proportion")

# Parse arguments
args = parser.parse_args()

# Access parsed arguments
print("Job Name:", args.job_name)
print("Seed:", args.seed)
print("Model Type:", args.ModelType)


############################################################################################################################################

WORK ABOVE 

############################################################################################################################################


### Set Up ###
ErrorVecSimulation = []
HistoryVecSimulation = []

### Run Code ###
ErrorVec, HistoryVec = OneIterationFunction(DataFileInput = ParameterVector.iloc[i]["Data"],
                                                Seed = ParameterVector.iloc[i]["Seed"],
                                                TestProportion = ParameterVector.iloc[i]["TestProportion"],
                                                CandidateProportion = ParameterVector.iloc[i]["CandidateProportion"],
                                                SelectorType = globals().get(ParameterVector.iloc[i]["SelectorType"], None), 
                                                ModelType = globals().get(ParameterVector.iloc[i]["ModelType"], None), 
                                                DataArgs = ast.literal_eval(ParameterVector.iloc[i]["DataArgs"]),
                                                SelectorArgs = ast.literal_eval(ParameterVector.iloc[i]["SelectorArgs"].replace("[","").replace("]","")),
                                                ModelArgs = ast.literal_eval(ParameterVector.iloc[i]["ModelArgs"])
                                                )

### Save Error Vec ###
ErrorVec.to_csv('out.csv', index=False) 