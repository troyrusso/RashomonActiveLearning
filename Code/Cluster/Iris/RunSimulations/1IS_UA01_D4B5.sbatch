#!/bin/bash
#SBATCH --job-name=1IS_UA01_D4B5
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_1IS_UA01_D4B5_%j.out
#SBATCH -e ClusterMessages/error/myscript_1IS_UA01_D4B5_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 1IS_UA01_D4B5 \
    --Data Iris \
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
    --BatchSize 5 \
    --DiversityWeight 0.4 \
    --Output Iris/TreeFarms/Raw/1IS_UA01_D4B5.pkl