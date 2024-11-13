### Import packages ###
import numpy as np
import matplotlib.pyplot as plt

### Function ###
def MeanVariancePlot(Subtitle = None,
                     **SimulationErrorResults):

    ### Set Up ###
    MeanVector = {}
    VarianceVector = {}

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis =0)
        VarianceVector[Label] = np.var(Results, axis =0)

    ### Mean Plot ###
    plt.figure(figsize=(7, 6))
    for Label, MeanValues in MeanVector.items():
        plt.plot(range(len(MeanValues)), MeanValues, label=Label)
    plt.suptitle("Active Learning Mean RMSE Plot")
    plt.xlabel("Number of labelled observations")
    plt.ylabel("Mean RMSE")
    plt.title(Subtitle, fontsize = 9)
    plt.legend()
    MeanPlot = plt.gcf()
    
    ### Variance Plot ###
    plt.figure(figsize=(7, 6))
    for Label, VarianceValues in VarianceVector.items():
        plt.plot(range(len(VarianceValues)), VarianceValues, label=Label)
    plt.suptitle("Active Learning Variance of RMSE Plot")
    plt.xlabel("Number of labelled observations")
    plt.ylabel("Variance of RMSE")
    plt.title(Subtitle, fontsize = 9)
    VariancePlot = plt.gcf()

    return MeanPlot, VariancePlot