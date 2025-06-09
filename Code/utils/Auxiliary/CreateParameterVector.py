# utils/Auxiliary/CreateParameterVector.py

### Import packages ###
import itertools
import pandas as pd
import numpy as np

# Data: Iris  MONK1  MONK3  Bar7 (10)  COMPAS (50) | BankNote (10)  BreastCancer (5)  CarEvaluation (10)  FICO (50)  Haberman
def CreateParameterVectorFunction(Data,
                                  Seed,                     # range(0,50)
                                  RashomonThreshold,        # For TreeFarms
                                  DiversityWeight,          # For BatchQBC
                                  DensityWeight,            # For BatchQBC
                                  BatchSize,                # For all batch selectors
                                  Partition,                # SLURM partition
                                  Time,                     # SLURM time limit
                                  Memory,                   # SLURM memory limit
                                  IncludePL_RF=False,       # Passive Learning with RandomForestClassifierPredictor
                                  IncludePL_GPC=False,      # Passive Learning with GaussianProcessClassifierPredictor
                                  IncludePL_BNN=False,      # Passive Learning with BayesianNeuralNetworkPredictor
                                  IncludeBALD_BNN=False,    # BALD with BayesianNeuralNetworkPredictor
                                  IncludeBALD_GPC=False,    # BALD with GaussianProcessClassifierPredictor
                                  IncludeQBC_TreeFarms_Unique=False, # BatchQBC with TreeFarmsPredictor (UniqueErrorsInput=1)
                                  IncludeQBC_TreeFarms_Duplicate=False, # BatchQBC with TreeFarmsPredictor (UniqueErrorsInput=0)
                                  IncludeQBC_RF=False,      # BatchQBC with RandomForestClassifierPredictor
                                  IncludeLFR_TreeFarms=False, # NEW: For TreefarmsLFRPredictor (requires RefitFrequency)
                                  RefitFrequency=1          # Default refit frequency for LFR (1 = every iter)
                                  ):

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

    # Parameter Dictionary #
    all_parameter_dicts = []

    ### Base Parameter Dictionary ###
    base_params = {
        "Data": [Data],
        "Seed": list(Seed),
        "TestProportion": [0.2],
        "CandidateProportion": [0.8],
        "SelectorType": ["BatchQBCSelector"], 
        "ModelType": ["TreeFarmsPredictor"],        
        "UniqueErrorsInput": [0],
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
        "Memory": [Memory]
    }

    ### 1. PassiveLearningSelector and RandomForestClassifierPredictor ###
    if IncludePL_RF:
        PL_RF_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"],
            "ModelType": ["RandomForestClassifierPredictor"],
            "UniqueErrorsInput": [0], 
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
            "Memory": ["1000M"] 
        }
        all_parameter_dicts.append(PL_RF_ParameterDictionary)

    ### 2. PassiveLearningSelector and GaussianProcessClassifierPredictor ###
    if IncludePL_GPC:
        PL_GPC_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"],
            "ModelType": ["GaussianProcessClassifierPredictor"],
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
            "Memory": [Memory], 
            "kernel_type": ['RBF'],
            "kernel_length_scale": [1.0],
            "kernel_nu": [1.5],
            "optimizer": ['fmin_l_bfgs_b'],
            "n_restarts_optimizer": [0],
            "max_iter_predict": [100]
        }
        all_parameter_dicts.append(PL_GPC_ParameterDictionary)

    ### 3. PassiveLearningSelector and BayesianNeuralNetworkPredictor ###
    if IncludePL_BNN:
        PL_BNN_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"],
            "ModelType": ["BayesianNeuralNetworkPredictor"],
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
            "Memory": [Memory],
            "hidden_size": [50],
            "dropout_rate": [0.2],
            "epochs": [100],
            "learning_rate": [0.001],
            "batch_size_train": [32],
            "K_BALD_Samples": [20] 
        }
        all_parameter_dicts.append(PL_BNN_ParameterDictionary)

    ### 4. BALDSelector and BayesianNeuralNetworkPredictor ###
    if IncludeBALD_BNN:
        BALD_BNN_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BALDSelector"],
            "ModelType": ["BayesianNeuralNetworkPredictor"],
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
            "Memory": [Memory],
            "hidden_size": [50],
            "dropout_rate": [0.2],
            "epochs": [100],
            "learning_rate": [0.001],
            "batch_size_train": [32],
            "K_BALD_Samples": [20] 
        }
        all_parameter_dicts.append(BALD_BNN_ParameterDictionary)

    ### 5. BALDSelector and GaussianProcessClassifierPredictor ###
    if IncludeBALD_GPC:
        BALD_GPC_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BALDSelector"],
            "ModelType": ["GaussianProcessClassifierPredictor"],
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
            "Memory": [Memory],
            "kernel_type": ['RBF'],
            "kernel_length_scale": [1.0],
            "kernel_nu": [1.5],
            "optimizer": ['fmin_l_bfgs_b'],
            "n_restarts_optimizer": [0],
            "max_iter_predict": [100],
            "K_BALD_Samples": [20] 
        }
        all_parameter_dicts.append(BALD_GPC_ParameterDictionary)

    ### 6. BatchQBCSelector and TreeFarmsPredictor with UniqueErrorsInput=1 (UNREAL) ###
    if IncludeQBC_TreeFarms_Unique:
        QBC_TF_Unique_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"],
            "ModelType": ["TreeFarmsPredictor"],
            "UniqueErrorsInput": [1], 
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
            "Memory": [Memory]
        }
        all_parameter_dicts.append(QBC_TF_Unique_ParameterDictionary)

    ### 7. BatchQBCSelector and TreeFarmsPredictor with UniqueErrorsInput=0 (DUREAL) ###
    if IncludeQBC_TreeFarms_Duplicate:
        QBC_TF_Duplicate_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], 
            "ModelType": ["TreeFarmsPredictor"],
            "UniqueErrorsInput": [0], 
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
            "Memory": [Memory]
        }
        all_parameter_dicts.append(QBC_TF_Duplicate_ParameterDictionary)

    ### 8. BatchQBCSelector with RandomForestClassifierPredictor ###
    if IncludeQBC_RF:
        QBC_RF_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"],
            "ModelType": ["RandomForestClassifierPredictor"], 
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
            "Memory": ["1000M"]
        }
        all_parameter_dicts.append(QBC_RF_ParameterDictionary)

    # NEW: Include LFR TreeFarms (LFRPredictor)
    if IncludeLFR_TreeFarms:
        LFR_TF_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], 
            "ModelType": ["LFRPredictor"], 
            "UniqueErrorsInput": [0], 
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
            "Memory": [Memory],
            "RefitFrequency": [RefitFrequency] 
        }
        all_parameter_dicts.append(LFR_TF_ParameterDictionary)


    # Combine all parameter dictionaries into a single DataFrame
    if not all_parameter_dicts:
        return pd.DataFrame() 

    list_of_dfs = []
    for p_dict in all_parameter_dicts:
        list_of_dfs.append(pd.DataFrame.from_records(itertools.product(*p_dict.values()), columns=p_dict.keys()))
    
    ParameterVector = pd.concat(list_of_dfs, ignore_index=True)

    # Ensure all possible columns are present in the final DataFrame, filling NaNs for missing model-specific params
    all_possible_columns = sorted(list(set(col for d in all_parameter_dicts for col in d.keys())))
    ParameterVector = ParameterVector.reindex(columns=all_possible_columns)
    
    numeric_cols = ParameterVector.select_dtypes(include=np.number).columns
    ParameterVector[numeric_cols] = ParameterVector[numeric_cols].fillna(0)
    object_cols = ParameterVector.select_dtypes(include='object').columns
    ParameterVector[object_cols] = ParameterVector[object_cols].fillna('')


    # Sort and re-index after all concatenations
    ParameterVector = ParameterVector.sort_values("Seed")
    ParameterVector.index = range(0, ParameterVector.shape[0])

    ### Job and Output Name ###

    # Generate initial JobName string #
    ParameterVector["JobName"] = (
        ParameterVector["Seed"].astype(str) +
        ParameterVector["Data"].map(AbbreviationDictionary).astype(str) + 
        "_MT" + ParameterVector["ModelType"].astype(str) + 
        "_ST" + ParameterVector["SelectorType"].astype(str) + 
        "_UEI" + ParameterVector["UniqueErrorsInput"].astype(str) +
        ParameterVector["RashomonThresholdType"].astype(str) +
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

    # Conditionally add model-specific parameters to JobName for clarity without clutter    
    # BNN-specific parameters
    bnn_mask = ParameterVector["ModelType"] == "BayesianNeuralNetworkPredictor"
    if 'hidden_size' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_HS" + ParameterVector.loc[bnn_mask, "hidden_size"].astype(str)
    if 'dropout_rate' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_DR" + ParameterVector.loc[bnn_mask, "dropout_rate"].astype(str)
    if 'epochs' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_E" + ParameterVector.loc[bnn_mask, "epochs"].astype(str)
    if 'learning_rate' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_LR" + ParameterVector.loc[bnn_mask, "learning_rate"].astype(str)
    if 'batch_size_train' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_BST" + ParameterVector.loc[bnn_mask, "batch_size_train"].astype(str)
    
    # GPC-specific parameters
    gpc_mask = ParameterVector["ModelType"] == "GaussianProcessClassifierPredictor"
    if 'kernel_type' in ParameterVector.columns:
        ParameterVector.loc[gpc_mask, "JobName"] += "_KT" + ParameterVector.loc[gpc_mask, "kernel_type"].astype(str)
    if 'kernel_length_scale' in ParameterVector.columns:
        ParameterVector.loc[gpc_mask, "JobName"] += "_KLS" + ParameterVector.loc[gpc_mask, "kernel_length_scale"].astype(str)
    if 'kernel_nu' in ParameterVector.columns:
        ParameterVector.loc[gpc_mask, "JobName"] += "_KNU" + ParameterVector.loc[gpc_mask, "kernel_nu"].astype(str)

    # Parameters common to BNN and GPC (if K_BALD_Samples is passed to selector or model)
    if 'K_BALD_Samples' in ParameterVector.columns:
        bald_relevant_mask = (ParameterVector["ModelType"] == "BayesianNeuralNetworkPredictor") | \
                             (ParameterVector["ModelType"] == "GaussianProcessClassifierPredictor")
        ParameterVector.loc[bald_relevant_mask, "JobName"] += "_K" + ParameterVector.loc[bald_relevant_mask, "K_BALD_Samples"].astype(str)
    
    # LFR-specific parameter
    lfr_mask = ParameterVector["ModelType"] == "LFRPredictor" # <--- CHANGED TO SHORTER NAME
    if 'RefitFrequency' in ParameterVector.columns:
        ParameterVector.loc[lfr_mask, "JobName"] += "_RFREQ" + ParameterVector.loc[lfr_mask, "RefitFrequency"].astype(str)


    # Reorder and refine JobName abbreviations for new class names
    # Apply more specific/desired replacements first.
    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        # UNREAL/DUREAL for TreeFarms with QBC
        .str.replace(r"_MTTreeFarmsPredictor_STBatchQBCDiversitySelector_UEI1(A\d+)", r"_UNREAL_UEI1\1", regex=True) # Catches UEI1A...
        .str.replace(r"_MTTreeFarmsPredictor_STBatchQBCDiversitySelector_UEI0(A\d+)", r"_DUREAL_UEI0\1", regex=True) # Catches UEI0A...
        # PL_RF
        .str.replace(r"(_MTRandomForestClassifierPredictor_STPassiveLearningSelector_UEI0A0_DW0_DEW0)(_B\d+)", r"_PL_RF\2", regex=True)
        # PL_GPC
        .str.replace(r"(_MTGaussianProcessClassifierPredictor_STPassiveLearningSelector_UEI0A0_DW0_DEW0)(_B\d+)", r"_PL_GPC\2", regex=True)
        # PL_BNN
        .str.replace(r"(_MTBayesianNeuralNetworkPredictor_STPassiveLearningSelector_UEI0A0_DW0_DEW0)(_B\d+)", r"_PL_BNN\2", regex=True)
        # BALD_BNN
        .str.replace(r"(_MTBayesianNeuralNetworkPredictor_STBALDSelector_UEI0A0_DW0_DEW0)(_B\d+)", r"_BALD_BNN\2", regex=True)
        # BALD_GPC
        .str.replace(r"(_MTGaussianProcessClassifierPredictor_STBALDSelector_UEI0A0_DW0_DEW0)(_B\d+)", r"_BALD_GPC\2", regex=True)
        # QBC_RF 
        .str.replace(r"(_MTRandomForestClassifierPredictor_STBatchQBCDiversitySelector)(_UEI.*)", r"_QBC_RF\2", regex=True)
        # LFR specific 
        .str.replace(r"(_MTTreefarmsLFRPredictor_STBatchQBCDiversitySelector_UEI0_A0_RFREQ)(\d+)(_B\d+)", r"_LFR_RFREQ\2\3", regex=True) 
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
        ParameterVector["ModelType"].astype(str) + "/Raw/" +
        ParameterVector["JobName"] + ".pkl"
    )

    ### Find Missing Simulations (Optional) ###
    ### Return ###
    return ParameterVector

### FilterJobNames ###
def FilterJobNames(df, filter_strings):
    mask = df['JobName'].apply(lambda x: any(filter_str in x for filter_str in filter_strings))
    return df[mask]