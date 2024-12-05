#!/bin/bash
#SBATCH --job-name=Seed77_DataBostonHousingBinned_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum5
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed77_DataBostonHousingBinned_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum5_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed77_DataBostonHousingBinned_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum5_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName Seed77_DataBostonHousingBinned_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum5 \
    --Seed 77 \
    --Data BostonHousingBinned \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --DataArgs '{"N": 1000, "K": 5}' \
    --SelectorArgs '{"TopCModels": 5, "UniqueErrorsInput": 1}' \
    --ModelArgs '{"n_estimators": 100, "Seed": 1, "TopCModels": 5, "config": {"regularization": 0.01, "rashomon_bound_multiplier": 0.1}, "Type": "Classification"}' \
    --Output BostonHousingBinned/TreeFarms/Raw/Seed77_DataBostonHousingBinned_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum5.pkl