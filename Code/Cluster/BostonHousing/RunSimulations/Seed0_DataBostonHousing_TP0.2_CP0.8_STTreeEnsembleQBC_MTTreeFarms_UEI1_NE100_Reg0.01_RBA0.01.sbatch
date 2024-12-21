#!/bin/bash
#SBATCH --job-name=Seed0_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Seed0_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01_%j.out
#SBATCH -e ClusterMessages/error/myscript_Seed0_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName Seed0_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01 \
    --Data BostonHousing \
    --Seed 0 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --rashomon_bound_adder 0.01 \
    --Type Classification \
    --Output BostonHousing/TreeFarms/Raw/Seed0_DataBostonHousing_TP0.2_CP0.8_STTreeEnsembleQBC_MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01.pkl