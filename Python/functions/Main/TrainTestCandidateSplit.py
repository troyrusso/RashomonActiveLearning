### NOTE: Need to floor the proprtions ###

### Libraries ###
import pandas as pd
from sklearn.model_selection import train_test_split


### Function ###
def TrainTestCandidateSplit(df, TestProportion, CandidateProportion):
     # Train/Test split
    X_TrainCandidate, X_Test, y_TrainCandidate, y_Test = train_test_split(
        df.loc[:, df.columns != "Y"], df["Y"], test_size=TestProportion
    )
    # Train/Candidate split
    X_Train, X_Candidate, y_Train, y_Candidate = train_test_split(
        X_TrainCandidate, y_TrainCandidate, test_size=CandidateProportion
    )

    # Keep original column names
    df_Train = X_Train.copy()
    df_Train.insert(0, 'Y', y_Train)

    df_Test = X_Test.copy()
    df_Test.insert(0, 'Y', y_Test)

    df_Candidate = X_Candidate.copy()
    df_Candidate.insert(0, 'Y', y_Candidate)

    return df_Train, df_Test, df_Candidate