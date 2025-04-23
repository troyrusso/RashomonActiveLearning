### Import packages ###
import itertools
import pandas as pd
from utils.Auxiliary import FindMissingSimulations

# Data: Iris  MONK1  MONK3  Bar7 (10)  COMPAS (50) | BankNote (10)  BreastCancer (5)  CarEvaluation (10)  FICO (50)  Haberman
def CreateParameterVectorFunction(Data, 
                                  Seed,                     # range(0,50)
                                  RashomonThreshold,
                                  DiversityWeight,          # 0.4
                                  BatchSize,
                                  Partition,                # [short, medium, long, largemem, compute, cpu-g2-mem2x]
                                  Time,                     # [00:59:00, 11:59:00, 6-23:59:00]
                                  Memory,                   # [100M, 30000M, 100000M]
                                  IncludeRF = False,
                                  IncludePL = False):
    
    ### Data Abbreviations ###
    AbbreviationDictionary = {"BankNote": "BN",
                              "Bar7": "B7",
                              "BreastCancer": "BC",
                              "CarEvaluation": "CE",
                              "COMPAS": "CP",
                              "FICO": "FI",
                              "Haberman": "HM",
                              "Iris": "IS",
                              "MONK1": "M1",
                              "MONK3":"M3"}
    JobNameAbbrev = AbbreviationDictionary[Data]

    # Input Parameters #
    ParameterDictionary = {"Data":[Data],
                           "Seed":list(Seed),
                           #  "Seed":list([]),                       
                           "TestProportion":[0.2],
                           "CandidateProportion":[0.8],
                           "SelectorType":["BatchQBCDiversityFunction"],
                           "ModelType":["TreeFarmsFunction"],
                           "UniqueErrorsInput": [0,1],
                           "n_estimators": [100], 
                           "regularization": [0.01],
                           "RashomonThresholdType": ["Adder"],                                         # ["Adder", "Multiplier"]
                           "RashomonThreshold": [RashomonThreshold],
                           "Type": ["Classification"],
                           "DiversityWeight": [DiversityWeight],
                           "BatchSize": [BatchSize],
                           "Partition": [Partition],                                                        # [short, medium, long, largemem, compute, cpu-g2-mem2x]
                           "Time": [Time],                                                            # [00:59:00, 11:59:00, 6-23:59:00]
                           "Memory": [Memory]}                                                                # [100M, 30000M, 100000M]

    ### Create Parameter Vector ###
    ParameterVector = pd.DataFrame.from_records(itertools.product(*ParameterDictionary.values()), columns=ParameterDictionary.keys())


    ### Passive Learning ###
    if IncludePL:
        RandomForestParameterDictionary = {"Data":[Data],
                            "Seed":list(Seed),
                            "TestProportion":[0.2],
                            "CandidateProportion":[0.8],
                            "SelectorType":["PassiveLearning"],
                            "ModelType":["RandomForestClassificationFunction"],
                            "UniqueErrorsInput": [1],
                            "n_estimators": [100], 
                            "regularization": [0.01],
                            "RashomonThresholdType": ["Adder"],                                                    # ["Adder", "Multiplier"]
                            "RashomonThreshold": [0],
                            "Type": ["Classification"],
                            "DiversityWeight": [0],
                            "BatchSize": [BatchSize],
                            "Partition": [Partition],                                                        # [short, medium, long, largemem, or compute]
                            "Time": ["00:59:00"],                                                            # [00:59:00, 11:59:00, 6-23:59:00]
                            "Memory": [1000]}                                                                # [1000, 30000, 100000]

        RandomForestParameterVector = pd.DataFrame.from_records(itertools.product(*RandomForestParameterDictionary.values()), columns=RandomForestParameterDictionary.keys())
        ParameterVector = pd.concat([ParameterVector, RandomForestParameterVector]) # NOTE: Comment out to not include random forest baseline
        ParameterVector = ParameterVector.sort_values("Seed")
        ParameterVector.index = range(0, ParameterVector.shape[0])

    ### Include Random Forest ###
    if IncludeRF:
        RandomForestParameterDictionary = {"Data":[Data],
                            "Seed":list(Seed),
                            "TestProportion":[0.2],
                            "CandidateProportion":[0.8],
                            "SelectorType":["BatchQBCDiversityFunction"],
                            "ModelType":["RandomForestClassificationFunction"],
                            "UniqueErrorsInput": [0],
                            "n_estimators": [100], 
                            "regularization": [0.01],
                            "RashomonThresholdType": ["Adder"],                                                    # ["Adder", "Multiplier"]
                            "RashomonThreshold": [0],
                            "Type": ["Classification"],
                            "DiversityWeight": [DiversityWeight],
                            "BatchSize": [BatchSize],
                            "Partition": [Partition],                                                        # [short, medium, long, largemem, or compute]
                            "Time": ["00:59:00"],                                                            # [00:59:00, 11:59:00, 6-23:59:00]
                            "Memory": [1000]}                                                                # [1000, 30000, 100000]

        RandomForestParameterVector = pd.DataFrame.from_records(itertools.product(*RandomForestParameterDictionary.values()), columns=RandomForestParameterDictionary.keys())
        ParameterVector = pd.concat([ParameterVector, RandomForestParameterVector]) # NOTE: Comment out to not include random forest baseline
        ParameterVector = ParameterVector.sort_values("Seed")
        ParameterVector.index = range(0, ParameterVector.shape[0])

    ### Job and Output Name ###

    # Generate JobName #
    ParameterVector["JobName"] = (
    ParameterVector["Seed"].astype(str) +
    JobNameAbbrev + 
    "_MT" + ParameterVector["ModelType"].astype(str) +
    "_UEI" + ParameterVector["UniqueErrorsInput"].astype(str) +
    "_" + ParameterVector["RashomonThresholdType"].astype(str) + 
    ParameterVector["RashomonThreshold"].astype(str)+
    "_D" + ParameterVector["DiversityWeight"].astype(str) + 
    "B" + ParameterVector["BatchSize"].astype(str))

    # Replace Job Name #
    ParameterVector["JobName"] = (
    ParameterVector["JobName"]
    .str.replace(r"_MTTreeFarmsFunction_UEI0_", "_D", regex=True)
    .str.replace(r"_MTTreeFarmsFunction_UEI1_", "_U", regex=True)
    .str.replace(r"Adder", "A", regex=True)
    .str.replace(r"Multiplier", "M", regex=True)
    .str.replace(r"_MTRandomForestClassificationFunction_UEI0_", "_RF", regex=True)
    .str.replace(r"_MTRandomForestClassificationFunction_UEI1_", "_PL", regex=True)
    .str.replace(r"0.", "", regex=False)
    )

    # Output Name #
    ParameterVector["Output"] = ParameterVector["Data"].astype(str) + "/" + ParameterVector["ModelType"].astype(str) + "/Raw/" + ParameterVector["JobName"] + ".pkl"
    ParameterVector["Output"] = ParameterVector["Output"].str.replace("Function", "", regex=False)

    ### Return ###
    return ParameterVector

### FilterJobNames ###
def FilterJobNames(df, filter_strings):
    mask = df['JobName'].apply(lambda x: any(filter_str in x for filter_str in filter_strings))
    return df[mask]