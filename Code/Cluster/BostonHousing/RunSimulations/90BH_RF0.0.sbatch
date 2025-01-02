#!/bin/bash
#SBATCH --job-name=90BH_RF0.0
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_90BH_RF0.0_%j.out
#SBATCH -e ClusterMessages/error/myscript_90BH_RF0.0_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 90BH_RF0.0 \
    --Data BostonHousing \
    --Seed 90 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType RandomForestClassificationFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.0 \
    --rashomon_bound_adder 0.0 \
    --Type Classification \
    --Output BostonHousing/RandomForestClassification/Raw/90BH_RF0.0.pkl