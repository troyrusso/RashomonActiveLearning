# Summary:
# Input:
# Output:

### Import packages ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Function ###
def MeanVariancePlot(Subtitle = None,
                     TransparencyVal = 0.2,
                     CriticalValue = 1.96,
                     RelativeError = None,
                     **SimulationErrorResults):

    ### Set Up ###
    MeanVector = {}
    VarianceVector = {}
    StdErrorVector ={}
    Y_Label = "Error"

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis =0)
        VarianceVector[Label] = np.var(Results, axis =0)
        StdErrorVector[Label] = np.std(Results, axis=0) / np.sqrt(Results.shape[0])

    ### Normalize to Relative Error if specified ###
    if RelativeError:
        if RelativeError in MeanVector:
            Y_Label = "Mean Error relative to " + RelativeError
            BaselineMean = MeanVector[RelativeError]
            BaselineVariance = VarianceVector[RelativeError]
            for Label in MeanVector:
                MeanVector[Label] = pd.Series(MeanVector[Label].values / BaselineMean.values, index=MeanVector[Label].index)
                StdErrorVector[Label] = pd.Series(StdErrorVector[Label].values / BaselineMean.values, index=StdErrorVector[Label].index)
                VarianceVector[Label] = pd.Series(VarianceVector[Label].values / BaselineVariance.values, index=VarianceVector[Label].index)
        else:
            raise ValueError(f"RelativeError='{RelativeError}' not found in provided results.")


    ### Mean Plot ###
    plt.figure(figsize=(7, 6))
    for Label, MeanValues in MeanVector.items():
        StdErrorValues = StdErrorVector[Label]
        x = range(len(MeanValues))
        plt.plot(x, MeanValues, label=Label)
        plt.fill_between(x, MeanValues - CriticalValue * StdErrorValues, 
                        MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal)

    plt.suptitle("Active Learning Mean Error Plot")
    plt.xlabel("Number of labelled observations")
    plt.ylabel(Y_Label)
    plt.title(Subtitle, fontsize=9)
    plt.legend()
    MeanPlot = plt.gcf()

    ### Variance Plot ###
    plt.figure(figsize=(7, 6))
    for Label, VarianceValues in VarianceVector.items():
        plt.plot(range(len(VarianceValues)), VarianceValues, label=Label)
    plt.suptitle("Active Learning Variance of Error Plot")
    plt.xlabel("Number of labelled observations")
    plt.ylabel("Variance of " + Y_Label)
    plt.title(Subtitle, fontsize = 9)
    plt.legend()
    VariancePlot = plt.gcf()

    return MeanPlot, VariancePlot