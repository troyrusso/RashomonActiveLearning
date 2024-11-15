### Import packages ###
import numpy as np
import matplotlib.pyplot as plt

### Function ###
def MeanVariancePlot(Subtitle = None,
                     TransparencyVal = 0.2,
                     CriticalValue = 1.96,
                     **SimulationErrorResults):

    ### Set Up ###
    MeanVector = {}
    VarianceVector = {}
    StdErrorVector ={}

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis =0)
        VarianceVector[Label] = np.var(Results, axis =0)
        StdErrorVector[Label] = np.std(Results, axis=0) / np.sqrt(Results.shape[0])


    ### Mean Plot ###
    plt.figure(figsize=(7, 6))
    for Label, MeanValues in MeanVector.items():
        StdErrorValues = StdErrorVector[Label]
        x = range(len(MeanValues))
        plt.plot(x, MeanValues, label=Label)
        plt.fill_between(x, MeanValues - CriticalValue * StdErrorValues, 
                        MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal)

    plt.suptitle("Active Learning Mean RMSE Plot")
    plt.xlabel("Number of labelled observations")
    plt.ylabel("Mean RMSE")
    plt.title(Subtitle, fontsize=9)
    plt.legend()
    MeanPlot = plt.gcf()


    ####
    
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