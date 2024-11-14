#!/bin/bash
#SBATCH --job-name=ExtractResults_DataSimulate_STGSxFunction_MTRidgeRegression_extract
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/extract_ExtractResults_DataSimulate_STGSxFunction_MTRidgeRegression_%j.out
#SBATCH -e ClusterMessages/error/extract_ExtractResults_DataSimulate_STGSxFunction_MTRidgeRegression_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/ExtractError.py \
    --ResultsDirectory /Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/Simulations/ExtractResults \
    --DataType ['Simulate'] \
    --SelectorType GSxFunction \
    --ModelType RidgeRegression \