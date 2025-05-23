### Import packages ###
import itertools
import pandas as pd
from utils.Auxiliary import FindMissingSimulations

# Data: Iris  MONK1  MONK3  Bar7 (10)  COMPAS (50) | BankNote (10)  BreastCancer (5)  CarEvaluation (10)  FICO (50)  Haberman
def CreateParameterVectorFunction(Data,
                                  Seed,                     # range(0,50)
                                  RashomonThreshold,
                                  DiversityWeight,
                                  DensityWeight,
                                  BatchSize,
                                  Partition,                # [short, medium, long, largemem, compute, cpu-g2-mem2x]
                                  Time,                     # [00:59:00, 11:59:00, 6-23:59:00]
                                  Memory,                   # [100M, 30000M, 100000M]
                                  IncludeRF = False,
                                  IncludePL = False,
                                  IncludeBALD = False):

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

    ### Base Parameter Dictionary ###
    ParameterDictionary = {"Data":[Data],
                           "Seed":list(Seed),
                           "TestProportion":[0.2],
                           "CandidateProportion":[0.8],
                           "SelectorType":["BatchQBCDiversityFunction"],
                           "ModelType":["TreeFarmsFunction"],
                           "UniqueErrorsInput": [0,1],
                           "n_estimators": [100],
                           "regularization": [0.01],
                           "RashomonThresholdType": ["Adder"],
                           "RashomonThreshold": [RashomonThreshold],
                           "Type": ["Classification"],
                           "DiversityWeight": [DiversityWeight],
                           "DensityWeight": [DensityWeight],
                           "BatchSize": [BatchSize],
                           "Partition": [Partition],
                           "Time": [Time],
                           "Memory": [Memory]}


    ### Create Base Parameter Vector ###
    ParameterVector = pd.DataFrame.from_records(itertools.product(*ParameterDictionary.values()), columns=ParameterDictionary.keys())


    ### Passive Learning Baseline ###
    if IncludePL:
        PL_ParameterDictionary = {"Data":[Data],
                            "Seed":list(Seed),
                            "TestProportion":[0.2],
                            "CandidateProportion":[0.8],
                            "SelectorType":["PassiveLearning"],
                            "ModelType":["RandomForestClassificationFunction"],
                            "UniqueErrorsInput": [1], 
                            "n_estimators": [100],
                            "regularization": [0.01], 
                            "RashomonThresholdType": ["Adder"], 
                            "RashomonThreshold": [0], 
                            "Type": ["Classification"],
                            "DiversityWeight": [0], 
                            "DensityWeight": [0], 
                            "BatchSize": [BatchSize],
                            "Partition": [Partition],
                            "Time": ["00:59:00"],
                            "Memory": ["1000M"]}

        PL_ParameterVector = pd.DataFrame.from_records(itertools.product(*PL_ParameterDictionary.values()), columns=PL_ParameterDictionary.keys())
        ParameterVector = pd.concat([ParameterVector, PL_ParameterVector])


    ### Include Random Forest with QBC ###
    if IncludeRF:
        RF_ParameterDictionary = {"Data":[Data],
                            "Seed":list(Seed),
                            "TestProportion":[0.2],
                            "CandidateProportion":[0.8],
                            "SelectorType":["BatchQBCDiversityFunction"],
                            "ModelType":["RandomForestClassificationFunction"],
                            "UniqueErrorsInput": [0],
                            "n_estimators": [100],
                            "regularization": [0.01], 
                            "RashomonThresholdType": ["Adder"],
                            "RashomonThreshold": [0], 
                            "Type": ["Classification"],
                            "DiversityWeight": [DiversityWeight],
                            "DensityWeight": [DensityWeight],
                            "BatchSize": [BatchSize],
                            "Partition": [Partition],
                            "Time": ["00:59:00"],
                            "Memory": ["1000M"]}

        RF_ParameterVector = pd.DataFrame.from_records(itertools.product(*RF_ParameterDictionary.values()), columns=RF_ParameterDictionary.keys())
        ParameterVector = pd.concat([ParameterVector, RF_ParameterVector])


    ### Include BALD (Bayesian Neural Network) ###
    if IncludeBALD:
        # BNN_HIDDEN_SIZE = 50
        # BNN_DROPOUT_RATE = 0.2
        # BNN_EPOCHS = 100
        # BNN_LEARNING_RATE = 0.001
        # BNN_BATCH_SIZE_TRAIN = 32
        # BNN_K_BALD_SAMPLES = 20 # This is the K in log_probs_N_K_C

        BALD_ParameterDictionary = {"Data":[Data],
                                    "Seed":list(Seed),
                                    "TestProportion":[0.2],
                                    "CandidateProportion":[0.8],
                                    "SelectorType":["BaldSelectorFunction"], 
                                    "ModelType":["BayesianNeuralNetworkFunction"], 
                                    "UniqueErrorsInput": [0], 
                                    "n_estimators": [0], 
                                    "regularization": [0.0], 
                                    "RashomonThresholdType": ["Adder"], 
                                    "RashomonThreshold": [0], 
                                    "Type": ["Classification"],
                                    "DiversityWeight": [0], 
                                    "DensityWeight": [0], 
                                    "BatchSize": [BatchSize],
                                    "Partition": [Partition],
                                    "Time": [Time],
                                    "Memory": [Memory]}

        BALD_ParameterVector = pd.DataFrame.from_records(itertools.product(*BALD_ParameterDictionary.values()), columns=BALD_ParameterDictionary.keys())
        ParameterVector = pd.concat([ParameterVector, BALD_ParameterVector])


    # Sort and re-index after all concatenations
    ParameterVector = ParameterVector.sort_values("Seed")
    ParameterVector.index = range(0, ParameterVector.shape[0])

    ### Job and Output Name ###

    # Generate initial JobName string #
    ParameterVector["JobName"] = (
        ParameterVector["Seed"].astype(str) +
        JobNameAbbrev +
        "_MT" + ParameterVector["ModelType"].astype(str).str.replace("Function", "", regex=False) +
        "_ST" + ParameterVector["SelectorType"].astype(str).str.replace("Function", "", regex=False) +
        "_UEI" + ParameterVector["UniqueErrorsInput"].astype(str) +
        "_" + ParameterVector["RashomonThresholdType"].astype(str) +
        ParameterVector["RashomonThreshold"].astype(str) + 
        "_DW" + ParameterVector["DiversityWeight"].astype(str) +
        "_DEW" + ParameterVector["DensityWeight"].astype(str) +
        "_B" + ParameterVector["BatchSize"].astype(str) 
    )

    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"0\.(?=\d)", "", regex=True) 
        .str.replace(r"\.0(?!\d)", "", regex=True) 
    )

    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"(_MTBayesianNeuralNetwork_STBaldSelector).*(_B\d+)", r"_BALD\2", regex=True)
        .str.replace(r"(_MTRandomForestClassification_STPassiveLearning).*(_B\d+)", r"_PL\2", regex=True)
        .str.replace(r"_MTRandomForestClassification_STBatchQBCDiversity.*_DW(\d+)_DEW(\d+)(_B\d+)", r"_RF_DW\1_DEW\2\3", regex=True)
        .str.replace(r"_MTRandomForestClassification_STBatchQBCDiversity.*_DW0_DEW0(_B\d+)", r"_RF\1", regex=True)
        .str.replace(r"_MTTreeFarms_STBatchQBCDiversity_UEI0_", "_D", regex=True)
        .str.replace(r"_MTTreeFarms_STBatchQBCDiversity_UEI1_", "_U", regex=True)
    )

    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"Adder", "A", regex=True) 
        .str.replace(r"Multiplier", "M", regex=True) 
        .str.replace(r"__+", "_", regex=True) 
        .str.strip("_") 
    )

    # Output Name #
    ParameterVector["Output"] = (
        ParameterVector["Data"].astype(str) + "/" +
        ParameterVector["ModelType"].astype(str).str.replace("Function", "", regex=False) + "/Raw/" +
        ParameterVector["JobName"] + ".pkl"
    )

    ### Return ###
    return ParameterVector

### FilterJobNames ###
def FilterJobNames(df, filter_strings):
    mask = df['JobName'].apply(lambda x: any(filter_str in x for filter_str in filter_strings))
    return df[mask]