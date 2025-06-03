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

    ### Base Parameter Dictionary (Example: Default TreeFarms QBC) ###
    base_params = {
        "Data": [Data],
        "Seed": list(Seed),
        "TestProportion": [0.2],
        "CandidateProportion": [0.8],
        "SelectorType": ["BatchQBCDiversitySelector"], # Consistent class name
        "ModelType": ["TreeFarmsPredictor"],           # Consistent class name
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

    ### 1. PassiveLearningSelector and RandomForestClassifierPredictor ###
    if IncludePL_RF:
        PL_RF_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"],
            "ModelType": ["RandomForestClassifierPredictor"],
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
            "SelectorType": ["PassiveLearningSelector"],
            "ModelType": ["GaussianProcessClassifierPredictor"],
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
            "SelectorType": ["PassiveLearningSelector"],
            "ModelType": ["BayesianNeuralNetworkPredictor"],
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
            "SelectorType": ["BALDSelector"],
            "ModelType": ["BayesianNeuralNetworkPredictor"],
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
            "SelectorType": ["BALDSelector"],
            "ModelType": ["GaussianProcessClassifierPredictor"],
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

    ### 6. BatchQBCSelector and TreeFarmsPredictor with UniqueErrorsInput=1 (UNREAL) ###
    if IncludeQBC_TreeFarms_Unique:
        QBC_TF_Unique_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCDiversitySelector"],
            "ModelType": ["TreeFarmsPredictor"],
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

    ### 7. BatchQBCSelector and TreeFarmsPredictor with UniqueErrorsInput=0 (DUREAL) ###
    if IncludeQBC_TreeFarms_Duplicate:
        QBC_TF_Duplicate_ParameterDictionary = {
            "Data": [Data],
            "Seed": list(Seed),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCDiversitySelector"],
            "ModelType": ["TreeFarmsPredictor"],
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
            "SelectorType": ["BatchQBCDiversitySelector"], # Changed to class name
            "ModelType": ["RandomForestClassifierPredictor"], # Changed to class name
            "UniqueErrorsInput": [0], # Fixed to 0 for RF as it's not relevant
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
            "SelectorType": ["BatchQBCDiversitySelector"], # Or other selectors compatible with TreefarmsLFRPredictor
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
    if not all_parameter_dicts:
        return pd.DataFrame() # Return empty if no configs are selected

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

    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"0\.(?=\d)", "", regex=True) 
        .str.replace(r"\.0(?!\d)", "", regex=True) 
    )

    # Reorder and refine JobName abbreviations for new class names
    # Apply more specific/desired replacements first.
    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        # UNREAL/DUREAL for TreeFarms with QBC
        .str.replace(r"_MTTreeFarmsPredictor_STBatchQBCDiversitySelector_UEI1A", r"_UNREAL_UEI1A", regex=True) # Catches UEI1A...
        .str.replace(r"_MTTreeFarmsPredictor_STBatchQBCDiversitySelector_UEI0A", r"_DUREAL_UEI0A", regex=True) # Catches UEI0A...
        # PL_RF
        .str.replace(r"(_MTRandomForestClassifierPredictor_STPassiveLearningSelector)(.*_B\d+)", r"_PL_RF\2", regex=True) # Capture from batchsize
        # PL_GPC (already correct)
        .str.replace(r"(_MTGaussianProcessClassifierPredictor_STPassiveLearningSelector)(.*_B\d+)", r"_PL_GPC\2", regex=True)
        # PL_BNN (already correct)
        .str.replace(r"(_MTBayesianNeuralNetworkPredictor_STPassiveLearningSelector)(.*_B\d+)", r"_PL_BNN\2", regex=True)
        # BALD_BNN
        .str.replace(r"(_MTBayesianNeuralNetworkPredictor_STBALDSelector)(.*_B\d+)", r"_BALD_BNN\2", regex=True)
        # BALD_GPC (already correct)
        .str.replace(r"(_MTGaussianProcessClassifierPredictor_STBALDSelector)(.*_B\d+)", r"_BALD_GPC\2", regex=True)
        # QBC_RF (ensure this captures the rest of the string for specific parameters)
        # This needs to replace the MT...ST part, and leave the UEI...DW...DEW...B part.
        .str.replace(r"(_MTRandomForestClassifierPredictor_STBatchQBCDiversitySelector)(_UEI.*)", r"_QBC_RF\2", regex=True)
        # LFR specific (already correct, but ensure consistency with BatchQBCDiversitySelector if changed to BatchQBCSelector)
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
    # If you have a FindMissingSimulations function, you can call it here.
    # ParameterVector = FindMissingSimulations(ParameterVector)

    ### Return ###
    return ParameterVector

### FilterJobNames ###
def FilterJobNames(df, filter_strings):
    mask = df['JobName'].apply(lambda x: any(filter_str in x for filter_str in filter_strings))
    return df[mask]