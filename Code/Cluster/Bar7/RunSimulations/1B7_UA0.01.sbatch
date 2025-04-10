#!/bin/bash
#SBATCH --job-name=1B7_UA0.01
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_1B7_UA0.01_%j.out
#SBATCH -e ClusterMessages/error/myscript_1B7_UA0.01_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 1B7_UA0.01 \
    --Data Bar7 \
    --Seed 1 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.01 \
    --Type Classification \
    --Output Bar7/TreeFarms/Raw/1B7_UA0.01.pkl