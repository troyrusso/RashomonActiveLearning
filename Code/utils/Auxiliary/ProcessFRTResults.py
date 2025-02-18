### Import libraries ###
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

### Process Files in Batches Function ###
def ProcessBatch(files, batch_size=5):
    ### Initialize Storage ###
    ThresholdValuesStorage = []
    Epsilon_F1ScoreStorage = []
    Epsilon_ClassAccuracyStorage = []
    
    ### Initialize Running Sums for Mean Calculation ###
    ThresholdValues_Sum = None
    Epsilon_F1Score_Sum = None
    Epsilon_ClassAccuracy_Sum = None
    file_count = 0
    
    ### Process in Batches ###
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        ### Process Current Batch ###
        for file in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}"):
            try:
                ### Load and Immediately Process File ###
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    
                    # Update storage for CSV files
                    ThresholdValuesStorage.append(data["ThresholdValues"])
                    Epsilon_F1ScoreStorage.append(data["Epsilon_F1Score"])
                    Epsilon_ClassAccuracyStorage.append(data["Epsilon_ClassAccuracy"])
                    
                    # Update running sums for mean calculation
                    if ThresholdValues_Sum is None:
                        ThresholdValues_Sum = np.array(data["ThresholdValues"])
                        Epsilon_F1Score_Sum = np.array(data["Epsilon_F1Score"])
                        Epsilon_ClassAccuracy_Sum = np.array(data["Epsilon_ClassAccuracy"])
                    else:
                        ThresholdValues_Sum += np.array(data["ThresholdValues"])
                        Epsilon_F1Score_Sum += np.array(data["Epsilon_F1Score"])
                        Epsilon_ClassAccuracy_Sum += np.array(data["Epsilon_ClassAccuracy"])
                    
                    file_count += 1
                    
                    # Clear data from memory
                    del data
                    
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
    
    ### Calculate Means ###
    ThresholdValues_Mean = ThresholdValues_Sum / file_count
    Epsilon_F1Score_Mean = Epsilon_F1Score_Sum / file_count
    Epsilon_ClassAccuracy_Mean = Epsilon_ClassAccuracy_Sum / file_count
    
    ### Convert to DataFrames for CSV Storage ###
    ThresholdValuesDF = pd.DataFrame(ThresholdValuesStorage)
    Epsilon_F1ScoreDF = pd.DataFrame(Epsilon_F1ScoreStorage)
    Epsilon_ClassAccuracyDF = pd.DataFrame(Epsilon_ClassAccuracyStorage)
    
    return (ThresholdValuesDF, Epsilon_F1ScoreDF, Epsilon_ClassAccuracyDF,
            ThresholdValues_Mean, Epsilon_F1Score_Mean, Epsilon_ClassAccuracy_Mean)

### Main Function ###
def main():
    ### Parser ###
    parser = argparse.ArgumentParser(description="Aggregate simulation results.")
    parser.add_argument("--DataType", type=str, required=True, help="Type of data.")
    args = parser.parse_args()

    ### Set Up Directories ###
    cwd = os.getcwd()
    ResultsDirectory = os.path.join(cwd, "Results")
    OutputDirectory = os.path.join(ResultsDirectory, "OptimalThreshold")
    RawDirectory = os.path.join(ResultsDirectory, "OptimalThreshold", args.DataType, "Raw")
    ProcessedDirectory = os.path.join(OutputDirectory, args.DataType, "Processed")
    
    ### Create Processed Directory if it Doesn't Exist ###
    os.makedirs(ProcessedDirectory, exist_ok=True)

    ### Extract File Names ###
    CategoryFileNames = [os.path.join(RawDirectory, filename) 
                        for filename in os.listdir(RawDirectory) 
                        if filename.endswith('.pkl')]

    ### Process Files in Batches ###
    print(f"Processing {len(CategoryFileNames)} files for {args.DataType}...")
    (ThresholdValuesStorage, Epsilon_F1ScoreStorage, Epsilon_ClassAccuracyStorage,
     ThresholdValues_Mean, Epsilon_F1Score_Mean, Epsilon_ClassAccuracy_Mean) = ProcessBatch(
        CategoryFileNames, 
        batch_size=5
    )

    ### Save Full Results as CSV ###
    print(f"Saving full results for {args.DataType}...")
    ThresholdValuesStorage.to_csv(os.path.join(ProcessedDirectory, "ThresholdValuesStorage.csv"), index=False)
    Epsilon_F1ScoreStorage.to_csv(os.path.join(ProcessedDirectory, "Epsilon_F1ScoreStorage.csv"), index=False)
    Epsilon_ClassAccuracyStorage.to_csv(os.path.join(ProcessedDirectory, "Epsilon_ClassAccuracyStorage.csv"), index=False)

    ### Save Mean Values as Pickle ###
    print(f"Saving mean values for {args.DataType}...")
    MeanValues = {
        "ThresholdValues_Mean": ThresholdValues_Mean,
        "Epsilon_F1Score_Mean": Epsilon_F1Score_Mean,
        "Epsilon_ClassAccuracy_Mean": Epsilon_ClassAccuracy_Mean
    }
    with open(os.path.join(ProcessedDirectory, f"{args.DataType}MeanValues.pkl"), 'wb') as f:
        pickle.dump(MeanValues, f)

    print(f"Successfully saved {args.DataType} FRT files!")

### Run Main ###
if __name__ == "__main__":
    main()