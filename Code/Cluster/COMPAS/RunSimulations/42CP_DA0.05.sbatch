#!/bin/bash
#SBATCH --job-name=42CP_DA0.05
#SBATCH --partition=compute-hugemem
#SBATCH --ntasks=1
#SBATCH --time=2-11:59:00
#SBATCH --mem-per-cpu=200G
#SBATCH -o ClusterMessages/out/myscript_42CP_DA0.05_%j.out
#SBATCH -e ClusterMessages/error/myscript_42CP_DA0.05_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 42CP_DA0.05 \
    --Data COMPAS \
    --Seed 42 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.05 \
    --Type Classification \
    --Output COMPAS/TreeFarms/Raw/42CP_DA0.05.pkl