### Libraries ###
import pandas as pd

def PassiveLearning(df_Candidate):
    ### Passive Sampling ###
    QueryObservation = df_Candidate.sample(n=1)
    return(QueryObservation.index[0])