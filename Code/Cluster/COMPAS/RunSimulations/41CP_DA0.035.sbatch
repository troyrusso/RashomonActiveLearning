#!/bin/bash
#SBATCH --job-name=41CP_DA0.035
#SBATCH --partition=compute-hugemem
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=150G
#SBATCH -o ClusterMessages/out/myscript_41CP_DA0.035_%j.out
#SBATCH -e ClusterMessages/error/myscript_41CP_DA0.035_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 41CP_DA0.035 \
    --Data COMPAS \
    --Seed 41 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.035 \
    --Type Classification \
    --Output COMPAS/TreeFarms/Raw/41CP_DA0.035.pkl