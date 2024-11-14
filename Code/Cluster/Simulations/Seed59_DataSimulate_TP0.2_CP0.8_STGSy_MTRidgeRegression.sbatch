#!/bin/bash
#SBATCH --job-name=Seed59_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed59_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed59_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName Seed59_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression \
    --Seed 59 \
    --Data Simulate \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType GSyFunction \
    --ModelType RidgeRegressionFunction \
    --DataArgs '{"N": 1000, "K": 5}' \
    --SelectorArgs '{"df_Candidate": "df_Candidate"}' \
    --ModelArgs '{"alpha_val": 0.1}' \
    --Output /Seed59_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression.csv