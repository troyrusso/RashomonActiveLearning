#!/bin/bash
#SBATCH --job-name=Seed97_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed97_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed97_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
Rscript Code/RunSimulation.py \
    --JobName Seed97_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1 \
    --Seed 97 \
    --Data Simulate \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType GSyFunction \
    --ModelType RidgeRegressionFunction \
    --DataArgs '{"N": 1000, "K": 5}' \
    --SelectorArgs '{"df_Candidate": "df_Candidate"}' \
    --ModelArgs '{"alpha_val": 0.1}' \
    --Output /Seed97_DataSimulate_TP0.2_CP0.8_STGSy_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1.csv