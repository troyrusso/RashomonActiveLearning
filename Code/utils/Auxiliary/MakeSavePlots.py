# Summary:
# Input:
# Output:

### Import Functions
import os
import pandas as pd
from utils.Auxiliary import *
from scipy.stats import wilcoxon

### Function ###
def MakePlotFunctions(DataType, ModelType, PlotArgs, SaveInput = None):

    ### Get Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, ".."))
    ProcessedResultsDirectory = os.path.join(ParentDirectory, "Results", DataType, ModelType, "ProcessedResults")

    ### Data ###
    SimulationErrorResults = {}
    result_files = [f for f in os.listdir(ProcessedResultsDirectory) if f.endswith("_ErrorMatrix.csv")]
    result_paths = {os.path.splitext(f)[0]: os.path.join(ProcessedResultsDirectory, f) for f in result_files}

    for key, path in result_paths.items():
        clean_key = key.replace("_ErrorMatrix", "")
        SimulationErrorResults[clean_key] = pd.read_csv(path)


    ### Plot ###
    PlotSubtitle = (
        f"Model: {ModelType} | Data: {DataType} | Iterations: {SimulationErrorResults[list(SimulationErrorResults.keys())[0]].shape[0]}"
    )
    MeanPlot, VariancePlot = MeanVariancePlot(Subtitle = PlotSubtitle,
                                              TransparencyVal = PlotArgs["TransparencyVal"],
                                              CriticalValue = PlotArgs["CriticalValue"],
                                              RelativeError = PlotArgs["RelativeError"],
                                              **{key: value for key, value in SimulationErrorResults.items()})
    
    ### Wilcoxon Ranked Sign Test ###
    WRSTResults = WilcoxonRankSignedTest({
        key: np.mean(value, axis=0)
        for key, value in SimulationErrorResults.items()
    })
    
    ### Save ###
    if SaveInput is not None:
        ## Relative ##
        if(PlotArgs["RelativeError"] is None):
            MeanPlotPath = os.path.join(ParentDirectory, "ResearchUpdates", str(SaveInput), DataType + ModelType + "MeanPlot.png")
            VariancePlotPath = os.path.join(ParentDirectory, "ResearchUpdates", str(SaveInput), DataType + ModelType + "VariancePlot.png")
        ## Not Relative ##
        else:
            MeanPlotPath = os.path.join(ParentDirectory, "ResearchUpdates", str(SaveInput), DataType + ModelType + 
                                        "MeanPlot" + PlotArgs["RelativeError"] + "Relative"+  ".png")
            VariancePlotPath = os.path.join(ParentDirectory, "ResearchUpdates", str(SaveInput), DataType + ModelType + 
                                        "VariancePlot" + PlotArgs["RelativeError"] + "Relative" +  ".png")
        ### Save ###
        MeanPlot.savefig(MeanPlotPath)
        VariancePlot.savefig(VariancePlotPath)

    ### Return ###
    return WRSTResults, MeanPlot, VariancePlot
    