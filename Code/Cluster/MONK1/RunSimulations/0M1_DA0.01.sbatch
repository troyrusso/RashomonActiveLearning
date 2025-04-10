#!/bin/bash
#SBATCH --job-name=0M1_DA0.01
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_0M1_DA0.01_%j.out
#SBATCH -e ClusterMessages/error/myscript_0M1_DA0.01_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 0M1_DA0.01 \
    --Data MONK1 \
    --Seed 0 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.01 \
    --Type Classification \
    --Output MONK1/TreeFarms/Raw/0M1_DA0.01.pkl