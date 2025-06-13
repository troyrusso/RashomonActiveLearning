### Packages ###
import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
import os
import warnings

from utils.Auxiliary.LoadAnalyzedData import LoadAnalyzedData
from utils.Auxiliary.MeanVariancePlot import MeanVariancePlot

### Analyze Results Function ###
def AnalyzeResultsFunction(DataType, methods_to_plot=None):
    """
    Analyzes and plots results for various active learning simulation methods.

    Args:
        DataType (str): The name of the data set (e.g., "Iris").
        methods_to_plot (list, optional): A list of method keys to include in the
                                          trace plots. If None, all available methods
                                          are plotted. Defaults to None.
    """

    ### Load Data ###
    BaseDirectory = os.path.join(os.path.expanduser("~"), "Documents", "RashomonActiveLearning", "Results")

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
        data = LoadAnalyzedData(data_type=DataType, base_directory=BaseDirectory, model_directory=config["ModelDir"], file_prefix=config["FilePrefix"])
        loaded_data_by_method[method_key] = data
        raw_data_tables[method_key] = data 

        if data.get("Error") is None:
            warnings.warn(f"No 'Error' data loaded for {method_key}.")

    ### Shape Table ###
    ShapeTable = {key: data["Error"].shape[0] for key, data in loaded_data_by_method.items() if data.get("Error") is not None}
    ShapeTable = pd.DataFrame([ShapeTable]) if ShapeTable else pd.DataFrame()

    ### Time Table ###
    time_data_list = []
    for key in method_configs.keys():
        data = loaded_data_by_method[key]
        row_data = {"Method": key}
        
        if data.get("Time") is not None and not data["Time"].empty:
            time_in_seconds = data["Time"].iloc[:, 0]
            row_data["Mean (minutes)"] = float(f"{time_in_seconds.mean() / 60:.2f}")
            row_data["Max (minutes)"] = float(f"{time_in_seconds.max() / 60:.2f}")
        else:
            row_data["Mean (minutes)"] = np.nan
            row_data["Max (minutes)"] = np.nan
        time_data_list.append(row_data)

    if time_data_list:
        TimeTable = pd.DataFrame(time_data_list).set_index("Method")
    else:
        TimeTable = pd.DataFrame()

    ### Legend and Styling Definitions ###
    PlotSubtitle = f"Dataset: {DataType}"
    colors = {"RF_PL": "black", "GPC_PL": "gray", "BNN_PL": "silver", "BNN_BALD": "darkviolet", "GPC_BALD": "mediumorchid", "RF_QBC": "green", "UNREAL_LFR": "dodgerblue", "DUREAL_LFR": "darkorange"}
    linestyles = {method: "solid" for method in method_configs.keys()}
    LegendMapping = {"RF_PL": "Passive Learning (RF)", "GPC_PL": "Passive Learning (GPC)", "BNN_PL": "Passive Learning (BNN)", "BNN_BALD": "BALD (BNN)", "GPC_BALD": "BALD (GPC)", "RF_QBC": "QBC (RF)", "UNREAL_LFR": "UNREAL_LFR", "DUREAL_LFR": "DUREAL_LFR"}

    ### Trace Plot (F1 Score) ###
    if methods_to_plot is None:
        methods_to_include = [key for key, data in loaded_data_by_method.items() if data.get("Error") is not None]
    else:
        methods_to_include = []
        for method in methods_to_plot:
            if method in loaded_data_by_method and loaded_data_by_method[method].get("Error") is not None:
                methods_to_include.append(method)
            else:
                warnings.warn(f"Requested method '{method}' for F1 plot not found or has no 'Error' data. It will be skipped.")
    
    error_data_for_plot = {key: loaded_data_by_method[key]["Error"] for key in methods_to_include}

    TracePlotMean, TracePlotVariance = None, None
    if not error_data_for_plot:
        warnings.warn("No valid data to plot for F1 score after filtering.")
    else:
        TracePlotMean, TracePlotVariance = MeanVariancePlot(RelativeError=None, Colors=colors, LegendMapping=LegendMapping, Linestyles=linestyles, Y_Label="F1 Score", Subtitle=PlotSubtitle, TransparencyVal=0.05, VarInput=True, CriticalValue=1.96, **error_data_for_plot)
        if TracePlotMean: TracePlotMean.get_axes()[0].legend(loc="best"); plt.close(TracePlotMean)
        if TracePlotVariance: TracePlotVariance.get_axes()[0].legend(loc="best"); plt.close(TracePlotVariance)

    ### Refit Frequency Plot ###
    refit_plot_data = {}
    
    # MODIFIED: Define a specific list of methods for this plot only
    refit_methods_to_plot = ["UNREAL_LFR", "DUREAL_LFR"]

    # Iterate over the specific list for the refit plot
    for key in refit_methods_to_plot:
        if key in loaded_data_by_method and loaded_data_by_method[key].get("RefitDecision") is not None:
            refit_plot_data[key] = loaded_data_by_method[key]["RefitDecision"]
        else:
            warnings.warn(f"RefitDecision data for '{key}' not found. It will be skipped in the RefitFrequencyPlot.")

    RefitFrequencyPlot = None
    if not refit_plot_data:
        warnings.warn("No RefitDecision data found for UNREAL_LFR or DUREAL_LFR.")
    else:
        RefitFrequencyPlot = MeanVariancePlot(
            RelativeError=None, 
            Colors=colors, 
            LegendMapping=LegendMapping, 
            Linestyles=linestyles, 
            Y_Label="Refit Frequency", 
            Subtitle=f"Dataset: {DataType} - Refit Behavior",
            TransparencyVal=0.05, 
            VarInput=False,
            CriticalValue=1.96, 
            **refit_plot_data
        )
        if RefitFrequencyPlot:
            RefitFrequencyPlot.get_axes()[0].legend(loc="best")
            plt.close(RefitFrequencyPlot)

    ### Number of Trees Plots ###
    unreal_plot_data, unreal_plot_legend, unreal_plot_colors = {}, {}, {}
    TreePlot_UNREAL = None
    unreal_data = loaded_data_by_method.get("UNREAL_LFR")
    if unreal_data:
        if unreal_data.get("AllTreeCount") is not None and not unreal_data["AllTreeCount"].empty:
            unreal_plot_data["Total"] = np.log(unreal_data["AllTreeCount"].replace(0, 1))
            unreal_plot_legend["Total"] = "Total Trees"
            unreal_plot_colors["Total"] = "darkorange"
        if unreal_data.get("UniqueTreeCount") is not None and not unreal_data["UniqueTreeCount"].empty:
            unreal_plot_data["Unique"] = np.log(unreal_data["UniqueTreeCount"].replace(0, 1))
            unreal_plot_legend["Unique"] = "Unique Trees"
            unreal_plot_colors["Unique"] = "dodgerblue"
        if unreal_plot_data:
            TreePlot_UNREAL = MeanVariancePlot(RelativeError=None, Colors=unreal_plot_colors, LegendMapping=unreal_plot_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - UNREAL_LFR", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **unreal_plot_data)
            if TreePlot_UNREAL: TreePlot_UNREAL.get_axes()[0].legend(loc="best"); plt.close(TreePlot_UNREAL)

    dureal_plot_data, dureal_plot_legend, dureal_plot_colors = {}, {}, {}
    TreePlot_DUREAL = None
    dureal_data = loaded_data_by_method.get("DUREAL_LFR")
    if dureal_data:
        if dureal_data.get("AllTreeCount") is not None and not dureal_data["AllTreeCount"].empty:
            dureal_plot_data["Total"] = np.log(dureal_data["AllTreeCount"].replace(0, 1))
            dureal_plot_legend["Total"] = "Total Trees"
            dureal_plot_colors["Total"] = "darkorange"
        if dureal_data.get("UniqueTreeCount") is not None and not dureal_data["UniqueTreeCount"].empty:
            dureal_plot_data["Unique"] = np.log(dureal_data["UniqueTreeCount"].replace(0, 1))
            dureal_plot_legend["Unique"] = "Unique Trees"
            dureal_plot_colors["Unique"] = "dodgerblue"
        if dureal_plot_data:
            TreePlot_DUREAL = MeanVariancePlot(RelativeError=None, Colors=dureal_plot_colors, LegendMapping=dureal_plot_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - DUREAL_LFR", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **dureal_plot_data)
            if TreePlot_DUREAL: TreePlot_DUREAL.get_axes()[0].legend(loc="best"); plt.close(TreePlot_DUREAL)

    ### Output ###
    return {
        "TracePlotMean": TracePlotMean,
        "TracePlotVariance": TracePlotVariance,
        "RefitFrequencyPlot": RefitFrequencyPlot,
        "TreePlot_UNREAL": TreePlot_UNREAL,
        "TreePlot_DUREAL": TreePlot_DUREAL,
        "ShapeTable": ShapeTable,
        "TimeTable": TimeTable,
        "RawData": raw_data_tables
    }