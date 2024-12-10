# Summary:
# Input:
# Output:

### Import packages ###
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

### Function ###
def MeanVariancePlot(Subtitle = None,
                     TransparencyVal = 0.2,
                     CriticalValue = 1.96,
                     RelativeError = None,
                     Colors=None, 
                     **SimulationErrorResults):

    ### Set Up ###
    MeanVector = {}
    VarianceVector = {}
    StdErrorVector ={}
    StdErrorVarianceVector = {}
    Y_Label = "Classification error"

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis=0)
        VarianceVector[Label] = np.var(Results, axis=0)
        StdErrorVector[Label] = np.std(Results, axis=0) / np.sqrt(Results.shape[0])
        
        # Compute CI for variance
        n = Results.shape[0]  # Number of samples
        lower_chi2 = chi2.ppf(0.025, df=n-1)  # Lower bound of Chi-square
        upper_chi2 = chi2.ppf(0.975, df=n-1)  # Upper bound of Chi-square
        StdErrorVarianceVector[Label] = {
            "lower": (n-1) * VarianceVector[Label] / upper_chi2,
            "upper": (n-1) * VarianceVector[Label] / lower_chi2
        }

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
        x = 20 + (np.arange(len(MeanValues)) / len(MeanValues)) * 80  # Start at 20% and go to 100%
        color = Colors.get(Label, None) if Colors else None  # Get color for the key
        plt.plot(x, MeanValues, label=Label, color=color)
        plt.fill_between(x, MeanValues - CriticalValue * StdErrorValues, 
                         MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal, color=color)


    # plt.suptitle("Active Learning Mean Error Plot")
    plt.xlabel("Percent of labelled observations")
    plt.ylabel(Y_Label)
    plt.title(Subtitle, fontsize=9)
    plt.legend()
    MeanPlot = plt.gcf()

    # Variance Plot
    plt.figure(figsize=(7, 6))
    for Label, VarianceValues in VarianceVector.items():
        x = 20 + (np.arange(len(VarianceValues)) / len(VarianceValues)) * 80  # Start at 20% and go to 100%
        color = Colors.get(Label, None) if Colors else None  # Get color for the key
        plt.plot(x, VarianceValues, label=Label, color=color)
        lower_bound = StdErrorVarianceVector[Label]["lower"]
        upper_bound = StdErrorVarianceVector[Label]["upper"]
        plt.fill_between(x, lower_bound, upper_bound, alpha=TransparencyVal, color=color)
    # plt.suptitle("Active Learning Variance of Error Plot")
    plt.xlabel("Percent of labelled observations")
    plt.ylabel("Variance of " + Y_Label)
    plt.title(Subtitle, fontsize = 9)
    plt.legend()
    VariancePlot = plt.gcf()

    return MeanPlot, VariancePlot