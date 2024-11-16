# Summary:
# Input:
# Output:

### Import Functions
import os
import pandas as pd
from utils.Auxiliary import *
from scipy.stats import wilcoxon

### Function ###
def MakePlotFunctions(DataType, ModelType, PlotArgs, SaveInput = False):

    ### Get Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, ".."))

    ### Data ###
    SimulationErrorResultsPassive = pd.read_csv(os.path.join(ParentDirectory, "Results", DataType, ModelType, "ProcessedResults", "PassiveLearning_ErrorVec.csv"))
    SimulationErrorResultsGSx = pd.read_csv(os.path.join(ParentDirectory, "Results", DataType, ModelType, "ProcessedResults", "GSx_ErrorVec.csv"))
    SimulationErrorResultsGSy = pd.read_csv(os.path.join(ParentDirectory, "Results", DataType, ModelType, "ProcessedResults", "GSy_ErrorVec.csv"))
    SimulationErrorResultsiGS = pd.read_csv(os.path.join(ParentDirectory, "Results", DataType, ModelType, "ProcessedResults", "iGS_ErrorVec.csv"))

    ### Plot ###
    PlotSubtitle = "Model: "+ ModelType + " |   Data: "+ DataType + "    |   Iterations: "+ str(SimulationErrorResultsPassive.shape[0])
    MeanPlot, VariancePlot = MeanVariancePlot(Subtitle = PlotSubtitle,
                                              TransparencyVal = PlotArgs["TransparencyVal"],
                                              CriticalValue = PlotArgs["CriticalValue"],
                                              RelativeRMSE = PlotArgs["RelativeRMSE"],
                                              Passive = SimulationErrorResultsPassive,
                                              GSx = SimulationErrorResultsGSx,
                                              GSy = SimulationErrorResultsGSy,
                                              iGS = SimulationErrorResultsiGS)
    
    ### Wilcoxon Ranked Sign Test ###
    WRSTResults = WilcoxonRankSignedTest({"Passive": np.mean(SimulationErrorResultsPassive, axis =0),
                                          "GSx" : np.mean(SimulationErrorResultsGSx, axis =0),
                                          "GSy" : np.mean(SimulationErrorResultsGSy, axis =0),
                                          "iGS" : np.mean(SimulationErrorResultsiGS, axis =0)})
    
    ### Return ###
    # Path #
    if(PlotArgs["RelativeRMSE"] is None):
        MeanPlotPath = os.path.join(ParentDirectory, "ResearchUpdates", "Nov15Updates", DataType, ModelType, "MeanPlot.png")
        VariancePlotPath = os.path.join(ParentDirectory, "ResearchUpdates", "Nov15Updates", DataType, ModelType, "VariancePlot.png")
    else:
        MeanPlotPath = os.path.join(ParentDirectory, "ResearchUpdates", "Nov15Updates", DataType, ModelType, 
                                    "MeanPlot" + PlotArgs["RelativeRMSE"] + "Relative"+  ".png")
        VariancePlotPath = os.path.join(ParentDirectory, "ResearchUpdates", "Nov15Updates", DataType, ModelType, 
                                    "VariancePlot" + PlotArgs["RelativeRMSE"] + "Relative"+  ".png")
    
    # Save #
    if SaveInput is True:
        MeanPlot.savefig(MeanPlotPath)
        VariancePlot.savefig(VariancePlotPath)

    return WRSTResults, MeanPlot, VariancePlot
    