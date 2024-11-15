#!/bin/bash
#SBATCH --job-name=Seed5_DataBostonHousing_TP0.2_CP0.8_STGSy_MTLinearRegression
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed5_DataBostonHousing_TP0.2_CP0.8_STGSy_MTLinearRegression_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed5_DataBostonHousing_TP0.2_CP0.8_STGSy_MTLinearRegression_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName Seed5_DataBostonHousing_TP0.2_CP0.8_STGSy_MTLinearRegression \
    --Seed 5 \
    --Data BostonHousing \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType GSyFunction \
    --ModelType LinearRegressionFunction \
    --DataArgs '{"N": 1000, "K": 5}' \
    --SelectorArgs '{"df_Candidate": "df_Candidate"}' \
    --ModelArgs '{"alpha_val": 0.1, "n_estimators": 100, "Seed": 1}' \
    --Output BostonHousing/LinearRegression/Raw/Seed5_DataBostonHousing_TP0.2_CP0.8_STGSy_MTLinearRegression.pkl