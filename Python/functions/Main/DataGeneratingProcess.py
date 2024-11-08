### Libraries ###
import numpy as np
import pandas as pd

### Functions ###
def ZFunc(x): return (x-0.2) / 0.4
def RFunc(x): return (ZFunc(x))**3 - 2*ZFunc(x)

### Data Generating Process ###
def DataGeneratingProcess(N, K):

    ### Generate Covariates ###
    X = np.random.uniform(0,1, size = (N,K))

    ### Generate Response ###
    Y = np.apply_along_axis(lambda x: RFunc(x).sum(), 1, X)

    ### Generate df Frame ###
    df = pd.DataFrame(X, columns = [f'X{i+1}' for i in range(K)])
    df.insert(0, 'Y', Y)  # Insert Y as the first column

    return df