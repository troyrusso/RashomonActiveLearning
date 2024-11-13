#!/bin/bash
#SBATCH --job-name=Seed51_DataSimulate_TP0.2_CP0.8_STPassiveLearning_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed51_DataSimulate_TP0.2_CP0.8_STPassiveLearning_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed51_DataSimulate_TP0.2_CP0.8_STPassiveLearning_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning/Code
module load Python
python RunSimulation.py \
    --JobName Seed51_DataSimulate_TP0.2_CP0.8_STPassiveLearning_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1 \
    --Seed 51 \
    --Data Simulate \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType PassiveLearning \
    --ModelType RidgeRegressionFunction \
    --DataArgs '{"N": 1000, "K": 5}' \
    --SelectorArgs '{"df_Candidate": "df_Candidate"}' \
    --ModelArgs '{"alpha_val": 0.1}' \
    --Output /Seed51_DataSimulate_TP0.2_CP0.8_STPassiveLearning_MTRidgeRegression_'N'1000,'K'5_'alpha_val'0.1.csv