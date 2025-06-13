### Packages ###
import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
import os
import warnings

# --- Make sure the new LoadAnalyzedData is importable ---
# This might require adjusting your Python path or project structure.
# For example, if utils is in the parent directory:
# sys.path.append('..')
from utils.Auxiliary.LoadAnalyzedData import LoadAnalyzedData
from utils.Auxiliary.MeanVariancePlot import MeanVariancePlot

### Analyze Results Function ###
def AnalyzeResultsFunction(DataType, RashomonThreshold):
    """
    Analyzes and plots results for various active learning simulation methods
    based on the new directory structure.

    Args:
        DataType (str): The name of the data set (e.g., "Iris").
        RashomonThreshold (float): The Rashomon Threshold for labeling plots.
                                   Note: This is now only for labeling purposes.

    Returns:
        dict: A dictionary containing generated plots, summary tables, and raw data.
    """

    ### Load Data ###
    BaseDirectory = os.path.join(os.path.expanduser("~"), "Documents", "RashomonActiveLearning", "Results")

    # --- Updated Method Configurations ---
    # Maps a short key to the parameters needed by LoadAnalyzedData.
    # "ModelDir" is the parent folder (e.g., 'BayesianNeuralNetworkPredictor').
    # "FilePrefix" is the unique identifier in the filename (e.g., '_BNN_BALD').
    method_configs = {
        "RF_PL":      {"ModelDir": "RandomForestClassifierPredictor", "FilePrefix": "_RF_PL"},
        "RF_QBC":     {"ModelDir": "RandomForestClassifierPredictor", "FilePrefix": "_RF_QBC"},
        "GPC_PL":     {"ModelDir": "GaussianProcessClassifierPredictor", "FilePrefix": "_GPC_PL"},
        "GPC_BALD":   {"ModelDir": "GaussianProcessClassifierPredictor", "FilePrefix": "_GPC_BALD"},
        "BNN_PL":     {"ModelDir": "BayesianNeuralNetworkPredictor", "FilePrefix": "_BNN_PL"},
        "BNN_BALD":   {"ModelDir": "BayesianNeuralNetworkPredictor", "FilePrefix": "_BNN_BALD"},
        "UNREAL_LFR": {"ModelDir": "LFRPredictor", "FilePrefix": "_Ulfr"},
        "DUREAL_LFR": {"ModelDir": "LFRPredictor", "FilePrefix": "_Dlfr"},
    }

    loaded_data_by_method = {}
    raw_data_tables = {}
    for method_key, config in method_configs.items():
        print(f"Loading {method_key} results for {DataType}...")
        data = LoadAnalyzedData(
            data_type=DataType,
            base_directory=BaseDirectory,
            model_directory=config["ModelDir"],
            file_prefix=config["FilePrefix"]
        )
        loaded_data_by_method[method_key] = data
        raw_data_tables[method_key] = data # Store for output

        # Warning if core "Error" data is missing
        if data.get("Error") is None:
            warnings.warn(f"No 'Error' data loaded for {method_key}. This method will be excluded from plots.")

    ### Shape Table (Number of Simulation Runs) ###
    ShapeTable = {key: data["Error"].shape[0] for key, data in loaded_data_by_method.items() if data.get("Error") is not None}
    ShapeTable = pd.DataFrame([ShapeTable]) if ShapeTable else pd.DataFrame()

    ### Time Table ###
    TimeTable = {}
    for key, data in loaded_data_by_method.items():
        if data.get("Time") is not None:
            time_in_seconds = data["Time"].iloc[:, 0] # Assuming time is in the first column
            TimeTable[key + " Mean (min)"] = f"{time_in_seconds.mean() / 60:.2f}"
            TimeTable[key + " Max (min)"] = f"{time_in_seconds.max() / 60:.2f}"
        else:
            TimeTable[key + " Mean (min)"] = "N/A"
            TimeTable[key + " Max (min)"] = "N/A"
    TimeTable = pd.DataFrame([TimeTable]) if TimeTable else pd.DataFrame()

    ### Trace Plot (F1 Score) ###
    PlotSubtitle = f"Dataset: {DataType}"
    colors = {
        "RF_PL": "black", "GPC_PL": "gray", "BNN_PL": "silver",
        "BNN_BALD": "darkviolet", "GPC_BALD": "mediumorchid",
        "RF_QBC": "green",
        "UNREAL_LFR": "dodgerblue", "DUREAL_LFR": "darkorange"
    }
    linestyles = {method: "solid" for method in method_configs.keys()}
    LegendMapping = {
        "RF_PL": "Passive Learning (RF)", "GPC_PL": "Passive Learning (GPC)", "BNN_PL": "Passive Learning (BNN)",
        "BNN_BALD": "BALD (BNN)", "GPC_BALD": "BALD (GPC)",
        "RF_QBC": "QBC (RF)",
        "UNREAL_LFR": f"UNREAL_LFR (ε = {RashomonThreshold})",
        "DUREAL_LFR": f"DUREAL_LFR (ε = {RashomonThreshold})"
    }

    # Dynamically prepare error data for plotting
    error_data_for_plot = {key: data["Error"] for key, data in loaded_data_by_method.items() if data.get("Error") is not None}

    TracePlotMean, TracePlotVariance = None, None
    if not error_data_for_plot:
        warnings.warn("No error data available for any method. Skipping TracePlot.")
    else:
        TracePlotMean, TracePlotVariance = MeanVariancePlot(
            RelativeError=None, Colors=colors, LegendMapping=LegendMapping, Linestyles=linestyles,
            Y_Label="F1 Score", Subtitle=PlotSubtitle, TransparencyVal=0.05,
            VarInput=True, CriticalValue=1.96, **error_data_for_plot
        )
        plt.close(TracePlotMean)
        plt.close(TracePlotVariance)

    ### Number of Trees Plot (Dynamic) ###
    tree_count_data = {}
    tree_legend = {}
    tree_colors = {}
    
    for key, data in loaded_data_by_method.items():
        if data.get("AllTreeCount") is not None:
            plot_key_all = f"{key}_All"
            tree_count_data[plot_key_all] = np.log(data["AllTreeCount"].replace(0, 1)) # Use log(1) for 0 trees
            tree_legend[plot_key_all] = f"{LegendMapping.get(key, key)} (Total)"
            tree_colors[plot_key_all] = "darkorange"
        if data.get("UniqueTreeCount") is not None:
            plot_key_unique = f"{key}_Unique"
            tree_count_data[plot_key_unique] = np.log(data["UniqueTreeCount"].replace(0, 1)) # Use log(1) for 0 trees
            tree_legend[plot_key_unique] = f"{LegendMapping.get(key, key)} (Unique)"
            tree_colors[plot_key_unique] = "dodgerblue"

    TreePlot = None
    if not tree_count_data:
        warnings.warn("No tree count data found for any method. Skipping TreePlot.")
    else:
        TreePlot = MeanVariancePlot(
            RelativeError=None, Colors=tree_colors, LegendMapping=tree_legend,
            Linestyles={key: 'solid' for key in tree_count_data},
            Y_Label="log(Number of Trees in Rashomon Set)", Subtitle=PlotSubtitle,
            TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **tree_count_data
        )
        plt.close(TreePlot)

    ### Wilcoxon Ranked Signed Test ###
    # WRSTResults = ... # This can be implemented next

    ### Output ###
    return {
        "TracePlotMean": TracePlotMean,
        "TracePlotVariance": TracePlotVariance,
        "TreePlot": TreePlot,
        "ShapeTable": ShapeTable,
        "TimeTable": TimeTable,
        "RawData": raw_data_tables # Include all loaded dataframes
    }