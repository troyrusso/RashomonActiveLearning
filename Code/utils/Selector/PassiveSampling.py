# Summary: Chooses an index at random from the candidate set to be queried.
# Input:
#   df_Candidate: The candidate set.
#   BatchSize: The number of observations to recommend.
#   auxiliary_columns: (Optional) A list of column names that are not features. (Added for compatibility)
# Output:
#   IndexRecommendation: The index of the recommended observation from the candidate set to be queried.

### Libraries ###

def PassiveLearning(df_Candidate, BatchSize=5, auxiliary_columns=None, **kwargs):
    if df_Candidate.shape[0] >= BatchSize:
         QueryObservation = df_Candidate.sample(n=BatchSize, random_state=kwargs.get('Seed', None)) 
         IndexRecommendation = list(QueryObservation.index)
    else:
        IndexRecommendation = list(df_Candidate.index)

    ### Output ###
    Output = {"IndexRecommendation": IndexRecommendation}
    return Output 