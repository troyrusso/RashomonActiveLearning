# utils/Auxiliary/CreateParameterVector.py (assuming this is the file)

### Import packages ###
import itertools
import pandas as pd
import numpy as np
# from utils.Auxiliary import FindMissingSimulations # Assuming this is a separate file/function

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

    # Initialize an empty list to store all parameter dictionaries
    all_parameter_dicts = []

    ### Base Parameter Dictionary (Example: Default TreeFarms QBC) ###
    # This can be set to whatever your primary simulation type is.
    # For now, let's make it flexible or remove it if all configs are driven by Include flags.
    # Let's assume the "base" is QBC with TreeFarms (UniqueErrorsInput=0) if no other flags are set.
    base_params = {
        "Data": [Data],
        "Seed": list(Seed),
        "TestProportion": [0.2],
        "CandidateProportion": [0.8],
        "SelectorType": ["BatchQBCSelector"], # Changed to class name
        "ModelType": ["TreeFarmsPredictor"],           # Changed to class name
        "UniqueErrorsInput": [0],
        "n_estimators": [100], # Keep for consistency, though TreeFarms doesn't use it
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
    # Add base_params to the list if you want it to always be included or if it's the default
    # For now, let's build all configs via the Include flags for clarity.
    # ParameterVector = pd.DataFrame.from_records(itertools.product(*base_params.values()), columns=base_params.keys())
    # all_parameter_dicts.append(base_params) # If you want this as a default run

    ### 1. PassiveLearningSelector and RandomForestClassifierPredictor ###
    if IncludePL_RF:
        PL_RF_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"], # Changed to class name
            "ModelType": ["RandomForestClassifierPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Not used by PL or RF, but keep for consistency
            "n_estimators": [100],
            "regularization": [0.01], # Not used by RF, but keep for consistency
            "RashomonThresholdType": ["Adder"], # Not used by RF, but keep for consistency
            "RashomonThreshold": [0], # Not used by RF, but keep for consistency
            "Type": ["Classification"],
            "DiversityWeight": [0], # Not used by PL, but keep for consistency
            "DensityWeight": [0], # Not used by PL, but keep for consistency
            "BatchSize": [BatchSize],
            "Partition": [Partition],
            "Time": ["00:59:00"], # Default time for simpler runs
            "Memory": ["1000M"] # Default memory for simpler runs
        }
        all_parameter_dicts.append(PL_RF_ParameterDictionary)

    ### 2. PassiveLearningSelector and GaussianProcessClassifierPredictor ###
    if IncludePL_GPC:
        PL_GPC_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"], # Changed to class name
            "ModelType": ["GaussianProcessClassifierPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Not used by PL or GPC
            "n_estimators": [0], # Not used by GPC
            "regularization": [0.0], # Not used by GPC
            "RashomonThresholdType": ["Adder"], # Not used by GPC
            "RashomonThreshold": [0], # Not used by GPC
            "Type": ["Classification"],
            "DiversityWeight": [0], # Not used by PL
            "DensityWeight": [0], # Not used by PL
            "BatchSize": [BatchSize],
            "Partition": [Partition],
            "Time": [Time], # GPC can be slow, use provided Time
            "Memory": [Memory], # GPC can be memory intensive, use provided Memory
            # GPC specific parameters
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
            "SelectorType": ["PassiveLearningSelector"], # Changed to class name
            "ModelType": ["BayesianNeuralNetworkPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Not used by PL or BNN
            "n_estimators": [0], # Not used by BNN
            "regularization": [0.0], # Not used by BNN
            "RashomonThresholdType": ["Adder"], # Not used by BNN
            "RashomonThreshold": [0], # Not used by BNN
            "Type": ["Classification"],
            "DiversityWeight": [0], # Not used by PL
            "DensityWeight": [0], # Not used by PL
            "BatchSize": [BatchSize],
            "Partition": [Partition],
            "Time": [Time],
            "Memory": [Memory],
            # BNN specific parameters
            "hidden_size": [50],
            "dropout_rate": [0.2],
            "epochs": [100],
            "learning_rate": [0.001],
            "batch_size_train": [32],
            "K_BALD_Samples": [20] # This is used by BNN's predict_proba_K, even if not for BALD selector
        }
        all_parameter_dicts.append(PL_BNN_ParameterDictionary)

    ### 4. BALDSelector and BayesianNeuralNetworkPredictor ###
    if IncludeBALD_BNN:
        BALD_BNN_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BALDSelector"], # Changed to class name
            "ModelType": ["BayesianNeuralNetworkPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Not used by BALD or BNN
            "n_estimators": [0], # Not used by BNN
            "regularization": [0.0], # Not used by BNN
            "RashomonThresholdType": ["Adder"], # Not used by BNN
            "RashomonThreshold": [0], # Not used by BNN
            "Type": ["Classification"],
            "DiversityWeight": [0], # Not used by BALD
            "DensityWeight": [0], # Not used by BALD
            "BatchSize": [BatchSize],
            "Partition": [Partition],
            "Time": [Time],
            "Memory": [Memory],
            # BNN specific parameters
            "hidden_size": [50],
            "dropout_rate": [0.2],
            "epochs": [100],
            "learning_rate": [0.001],
            "batch_size_train": [32],
            "K_BALD_Samples": [20] # This is used by BALDSelector
        }
        all_parameter_dicts.append(BALD_BNN_ParameterDictionary)

    ### 5. BALDSelector and GaussianProcessClassifierPredictor ###
    if IncludeBALD_GPC:
        BALD_GPC_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BALDSelector"], # Changed to class name
            "ModelType": ["GaussianProcessClassifierPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Not used by BALD or GPC
            "n_estimators": [0], # Not used by GPC
            "regularization": [0.0], # Not used by GPC
            "RashomonThresholdType": ["Adder"], # Not used by GPC
            "RashomonThreshold": [0], # Not used by GPC
            "Type": ["Classification"],
            "DiversityWeight": [0], # Not used by BALD
            "DensityWeight": [0], # Not used by BALD
            "BatchSize": [BatchSize],
            "Partition": [Partition],
            "Time": [Time],
            "Memory": [Memory],
            # GPC specific parameters
            "kernel_type": ['RBF'],
            "kernel_length_scale": [1.0],
            "kernel_nu": [1.5],
            "optimizer": ['fmin_l_bfgs_b'],
            "n_restarts_optimizer": [0],
            "max_iter_predict": [100],
            "K_BALD_Samples": [20] # This is used by BALDSelector
        }
        all_parameter_dicts.append(BALD_GPC_ParameterDictionary)

    ### 6. BatchQBCSelector and TreeFarmsPredictor with UniqueErrorsInput=1 ###
    if IncludeQBC_TreeFarms_Unique:
        QBC_TF_Unique_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], # Changed to class name
            "ModelType": ["TreeFarmsPredictor"], # Changed to class name
            "UniqueErrorsInput": [1], # Unique errors input for QBC
            "n_estimators": [100], # Keep for consistency, though TreeFarms doesn't use it
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

    ### 7. BatchQBCSelector and TreeFarmsPredictor with UniqueErrorsInput=0 ###
    if IncludeQBC_TreeFarms_Duplicate:
        QBC_TF_Duplicate_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], # Changed to class name
            "ModelType": ["TreeFarmsPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Duplicate errors input for QBC
            "n_estimators": [100], # Keep for consistency, though TreeFarms doesn't use it
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
            "SelectorType": ["BatchQBCSelector"], # Changed to class name
            "ModelType": ["RandomForestClassifierPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Not directly used by RF, but QBC might use it for committee pruning
            "n_estimators": [100],
            "regularization": [0.01], # Not used by RF
            "RashomonThresholdType": ["Adder"], # Not used by RF
            "RashomonThreshold": [0], # Not used by RF
            "Type": ["Classification"],
            "DiversityWeight": [DiversityWeight],
            "DensityWeight": [DensityWeight],
            "BatchSize": [BatchSize],
            "Partition": [Partition],
            "Time": ["00:59:00"],
            "Memory": ["1000M"]
        }
        all_parameter_dicts.append(QBC_RF_ParameterDictionary)

    # NEW: Include LFR TreeFarms (TreefarmsLFRPredictor)
    if IncludeLFR_TreeFarms:
        LFR_TF_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], # Or other selectors compatible with TreefarmsLFRPredictor
            "ModelType": ["TreefarmsLFRPredictor"], # The LFR model
            "UniqueErrorsInput": [0], # How QBC handles unique errors
            "n_estimators": [100], # Not used by TreefarmsLFRPredictor
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
            "RefitFrequency": [RefitFrequency] # Crucial for LFR
        }
        all_parameter_dicts.append(LFR_TF_ParameterDictionary)


    # Combine all parameter dictionaries into a single DataFrame
    # Use pd.DataFrame.from_dict for each dict, then concat
    # This approach is more flexible than itertools.product on a single large dict
    if not all_parameter_dicts:
        return pd.DataFrame() # Return empty if no configs are selected

    # Create a list of DataFrames, one for each config
    list_of_dfs = []
    for p_dict in all_parameter_dicts:
        # Ensure all keys are present in each dict for consistent columns
        # Fill missing keys with a default (e.g., None or 0) if they are not relevant to a specific model
        # This is important if you have model-specific params like 'kernel_type' that other models don't have.
        # For now, let's assume itertools.product will handle missing keys by creating columns for all.
        list_of_dfs.append(pd.DataFrame.from_records(itertools.product(*p_dict.values()), columns=p_dict.keys()))
    
    # Concatenate all generated DataFrames
    ParameterVector = pd.concat(list_of_dfs, ignore_index=True)

    # Ensure all possible columns are present in the final DataFrame, filling NaNs for missing model-specific params
    # This step is crucial if different parameter dictionaries have different sets of keys.
    all_possible_columns = sorted(list(set(col for d in all_parameter_dicts for col in d.keys())))
    ParameterVector = ParameterVector.reindex(columns=all_possible_columns)
    
    # Fill NaN values for parameters not relevant to a specific model/selector
    # For example, n_estimators might be NaN for BNN, fill with 0 or a placeholder.
    # This depends on how downstream functions handle these parameters.
    # A robust way is to fill NaNs for numeric parameters with 0 or a sensible default.
    numeric_cols = ParameterVector.select_dtypes(include=np.number).columns
    ParameterVector[numeric_cols] = ParameterVector[numeric_cols].fillna(0)
    # For string/object columns that might be NaN, fill with '' or 'N/A'
    object_cols = ParameterVector.select_dtypes(include='object').columns
    ParameterVector[object_cols] = ParameterVector[object_cols].fillna('')


    # Sort and re-index after all concatenations
    ParameterVector = ParameterVector.sort_values("Seed")
    ParameterVector.index = range(0, ParameterVector.shape[0])

    ### Job and Output Name ###

    # Generate initial JobName string #
    # Make sure to use the new class names for string replacement
    ParameterVector["JobName"] = (
        ParameterVector["Seed"].astype(str) +
        ParameterVector["Data"].map(AbbreviationDictionary).astype(str) + # Use map for abbreviation
        "_MT" + ParameterVector["ModelType"].astype(str) + # No more Function suffix
        "_ST" + ParameterVector["SelectorType"].astype(str) + # No more Function suffix
        "_UEI" + ParameterVector["UniqueErrorsInput"].astype(str) +
        ParameterVector["RashomonThresholdType"].astype(str) +
        ParameterVector["RashomonThreshold"].astype(str) + 
        "_DW" + ParameterVector["DiversityWeight"].astype(str) +
        "_DEW" + ParameterVector["DensityWeight"].astype(str) +
        "_B" + ParameterVector["BatchSize"].astype(str) 
    )

    # Add model-specific parameters to JobName if they exist and are not 0/empty
    # This makes JobName more descriptive for BNN, GPC, etc.
    if 'hidden_size' in ParameterVector.columns:
        ParameterVector["JobName"] += "_HS" + ParameterVector["hidden_size"].astype(str)
    if 'dropout_rate' in ParameterVector.columns:
        ParameterVector["JobName"] += "_DR" + ParameterVector["dropout_rate"].astype(str)
    if 'epochs' in ParameterVector.columns:
        ParameterVector["JobName"] += "_E" + ParameterVector["epochs"].astype(str)
    if 'learning_rate' in ParameterVector.columns:
        ParameterVector["JobName"] += "_LR" + ParameterVector["learning_rate"].astype(str)
    if 'batch_size_train' in ParameterVector.columns:
        ParameterVector["JobName"] += "_BST" + ParameterVector["batch_size_train"].astype(str)
    if 'K_BALD_Samples' in ParameterVector.columns:
        ParameterVector["JobName"] += "_K" + ParameterVector["K_BALD_Samples"].astype(str)
    if 'kernel_type' in ParameterVector.columns:
        ParameterVector["JobName"] += "_KT" + ParameterVector["kernel_type"].astype(str)
    if 'kernel_length_scale' in ParameterVector.columns:
        ParameterVector["JobName"] += "_KLS" + ParameterVector["kernel_length_scale"].astype(str)
    if 'kernel_nu' in ParameterVector.columns:
        ParameterVector["JobName"] += "_KNU" + ParameterVector["kernel_nu"].astype(str)
    if 'RefitFrequency' in ParameterVector.columns:
        ParameterVector["JobName"] += "_RFREQ" + ParameterVector["RefitFrequency"].astype(str)


    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"0\.(?=\d)", "", regex=True) 
        .str.replace(r"\.0(?!\d)", "", regex=True) 
    )

    # Update JobName abbreviations for new class names
    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"(_MTBayesianNeuralNetworkPredictor_STBALDSelector).*(_B\d+)", r"_BALD\2", regex=True)
        .str.replace(r"(_MTRandomForestClassifierPredictor_STPassiveLearningSelector).*(_B\d+)", r"_PL\2", regex=True)
        .str.replace(r"(_MTGaussianProcessClassifierPredictor_STPassiveLearningSelector).*(_B\d+)", r"_PL_GPC\2", regex=True) # New PL_GPC
        .str.replace(r"(_MTBayesianNeuralNetworkPredictor_STPassiveLearningSelector).*(_B\d+)", r"_PL_BNN\2", regex=True) # New PL_BNN
        .str.replace(r"(_MTGaussianProcessClassifierPredictor_STBALDSelector).*(_B\d+)", r"_BALD_GPC\2", regex=True) # New BALD_GPC
        .str.replace(r"_MTRandomForestClassifierPredictor_STBatchQBCSelector_DW(\d+)_DEW(\d+)(_B\d+)", r"_RF_DW\1_DEW\2\3", regex=True)
        .str.replace(r"_MTRandomForestClassifierPredictor_STBatchQBCSelector_DW0_DEW0(_B\d+)", r"_RF\1", regex=True)
        .str.replace(r"_MTTreeFarmsPredictor_STBatchQBCSelector_UEI0_", "_D", regex=True) # TreeFarmsPredictor UniqueErrorsInput=0
        .str.replace(r"_MTTreeFarmsPredictor_STBatchQBCSelector_UEI1_", "_U", regex=True) # TreeFarmsPredictor UniqueErrorsInput=1
        .str.replace(r"_MTTreefarmsLFRPredictor_STBatchQBCSelector_UEI0_A0_RFREQ(\d+)(_B\d+)", r"_LFR_RFREQ\1\2", regex=True) # LFR specific
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
        ParameterVector["ModelType"].astype(str) + "/Raw/" + # No more Function suffix
        ParameterVector["JobName"] + ".pkl"
    )

    ### Find Missing Simulations (Optional) ###
    # If you have a FindMissingSimulations function, you can call it here.
    # ParameterVector = FindMissingSimulations(ParameterVector)

    ### Return ###
    return ParameterVector

### FilterJobNames ###
def FilterJobNames(df, filter_strings):
    mask = df['JobName'].apply(lambda x: any(filter_str in x for filter_str in filter_strings))
    return df[mask]