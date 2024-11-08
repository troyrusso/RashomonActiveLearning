### NOTE: Need to floor the proprtions ###

### Libraries ###
import pandas as pd
from sklearn.model_selection import train_test_split


### Function ###
def TrainTestCandidateSplit(df, TestProportion, CandidateProportion):
    ### Train Test Candidate Split ###
    X_TrainCandidate, X_Test, y_TrainCandidate, y_Test = train_test_split(df.loc[:, df.columns != "Y"], df["Y"], test_size= TestProportion)
    X_Train, X_Candidate, y_Train, y_Candidate = train_test_split(X_TrainCandidate, y_TrainCandidate, test_size= CandidateProportion)

    ### Construct Data Frame ###
    # Train #
    df_Train = pd.DataFrame(X_Train, columns = [f'X{i+1}' for i in range(X_Train.shape[1])])
    df_Train.insert(0, 'Y', y_Train)

    # Test #
    df_Test = pd.DataFrame(X_Test, columns = [f'X{i+1}' for i in range(X_Test.shape[1])])
    df_Test.insert(0, 'Y', y_Test)

    # Candidate #
    df_Candidate = pd.DataFrame(X_Candidate, columns = [f'X{i+1}' for i in range(X_Candidate.shape[1])])
    df_Candidate.insert(0, 'Y', y_Candidate)

    return df_Train, df_Test, df_Candidate