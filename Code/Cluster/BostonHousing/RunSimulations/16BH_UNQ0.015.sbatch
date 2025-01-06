#!/bin/bash
#SBATCH --job-name=16BH_UNQ0.015
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_16BH_UNQ0.015_%j.out
#SBATCH -e ClusterMessages/error/myscript_16BH_UNQ0.015_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 16BH_UNQ0.015 \
    --Data BostonHousing \
    --Seed 16 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --rashomon_bound_adder 0.015 \
    --Type Classification \
    --Output BostonHousing/TreeFarms/Raw/16BH_UNQ0.015.pkl