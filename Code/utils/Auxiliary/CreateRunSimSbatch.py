### Import packages ###
import os
import numpy as np
import pandas as pd
import argparse

### Directory ###
cwd = os.getcwd()
ParentDirectory = os.path.abspath(os.path.join(cwd, "../.."))

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--DataType", type=str, default="-1", help="Simulation case number.")
args = parser.parse_args()

### Open Parameter Vector ###
ParameterVector = pd.read_csv(os.path.join(ParentDirectory, "Data", "ParameterVectors", "ParameterVector" + args.DataType + ".csv"))

# Loop through each row in the DataFrame
for i, row in ParameterVector.iterrows():
    # Extract parameters for the current row
    JobName = row["JobName"]
    Seed = row["Seed"]
    Data = row["Data"]
    TestProportion = row["TestProportion"]
    CandidateProportion = row["CandidateProportion"]
    SelectorType = row["SelectorType"]
    ModelType = row["ModelType"]
    DataArgs = row["DataArgs"]
    SelectorArgs = row["SelectorArgs"]
    ModelArgs = row["ModelArgs"]
    Output = row["Output"]
    
    # Define the path for the .sbatch file
    TargetDirectory = os.path.join(ParentDirectory,"Code", "Cluster", Data, "RunSimulations")
    sbatch_file_path = os.path.join(TargetDirectory, f"{JobName}.sbatch")
    
    # Create the .sbatch file content
    sbatch_content = [
        "#!/bin/bash",
        f"#SBATCH --job-name={JobName}",
        "#SBATCH --partition=short",
        "#SBATCH --ntasks=1",
        "#SBATCH --time=11:59:00",
        "#SBATCH --mem-per-cpu=30000",
        f"#SBATCH -o ClusterMessages/out/myscript_{JobName}_%j.out",
        f"#SBATCH -e ClusterMessages/error/myscript_{JobName}_%j.err",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=simondn@uw.edu",
        "",
        "cd ~/RashomonActiveLearning",
        "module load Python",
        "python Code/RunSimulation.py \\",
        f"    --JobName " + JobName +" \\",
        f"    --Seed {Seed} \\",
        f"    --Data {Data} \\",
        f"    --TestProportion {TestProportion} \\",
        f"    --CandidateProportion {CandidateProportion} \\",
        f"    --SelectorType {SelectorType} \\",
        f"    --ModelType {ModelType} \\",
        f"    --DataArgs {DataArgs} \\",
        f"    --SelectorArgs {SelectorArgs} \\",
        f"    --ModelArgs {ModelArgs} \\",
        f"    --Output {Output}"
    ]

    # Write content to .sbatch file
    os.makedirs(os.path.dirname(sbatch_file_path), exist_ok=True)  # Ensure directory exists
    with open(sbatch_file_path, "w") as sbatch_file:
        sbatch_file.write("\n".join(sbatch_content))

print("Creation Sbatch files generated successfully.")
