# Summary: Chooses an index at random from the candidate set to be queried.
# Input:
#   df_Candidate: The candidate set.
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

### Libraries ###
# import pandas as pd

def PassiveLearning(df_Candidate, BatchSize=5):

    if df_Candidate.shape[0] >= BatchSize:
         QueryObservation = df_Candidate.sample(n=BatchSize)
         IndexRecommendation = list(QueryObservation.index)
    else:
        IndexRecommendation = list(df_Candidate.index)

    ### Output ###
    Output = {"IndexRecommendation": IndexRecommendation}
    return(Output)