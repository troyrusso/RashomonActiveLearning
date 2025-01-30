#!/bin/bash
#SBATCH --job-name=27CE_UA0.07
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --ntasks=1
#SBATCH --time=2-11:59:00
#SBATCH --mem-per-cpu=150G
#SBATCH -o ClusterMessages/out/myscript_27CE_UA0.07_%j.out
#SBATCH -e ClusterMessages/error/myscript_27CE_UA0.07_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 27CE_UA0.07 \
    --Data CarEvaluation \
    --Seed 27 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.07 \
    --Type Classification \
    --Output CarEvaluation/TreeFarms/Raw/27CE_UA0.07.pkl