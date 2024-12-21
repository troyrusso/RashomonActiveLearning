#!/bin/bash
#SBATCH --job-name=Seed1_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed1_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed1_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName Seed1_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01 \
    --Data BostonHousing \
    --Seed 1 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType RandomForestClassificationFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --rashomon_bound_adder 0.01 \
    --Type Classification \
    --Output BostonHousing/RandomForestClassification/Raw/Seed1_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01.pkl