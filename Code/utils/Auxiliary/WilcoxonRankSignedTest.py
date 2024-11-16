# Summary:
# Input:
# Output:

### Packages ###
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

### Function ###
def WilcoxonRankSignedTest(SimulationErrorResults):

    ### Set Up ###
    strategies = list(SimulationErrorResults.keys())
    n_strategies = len(strategies)
    PValeMatrix = np.zeros((n_strategies, n_strategies))

    ### Wilcoxon Signed-Rank Test ###
    for i in range(n_strategies):
        for j in range(i):
            stat, pval = wilcoxon(SimulationErrorResults[strategies[i]], SimulationErrorResults[strategies[j]])
            PValeMatrix[i, j] = pval

    ### Formatting ###
    np.fill_diagonal(PValeMatrix, 1)
    pval_df = pd.DataFrame(PValeMatrix, index=strategies, columns=strategies)
    mask = np.tril(np.ones(pval_df.shape), k=0).astype(bool)
    WRSTResults = pval_df.where(mask, "").astype(str)

    ### Return ###
    return WRSTResults
