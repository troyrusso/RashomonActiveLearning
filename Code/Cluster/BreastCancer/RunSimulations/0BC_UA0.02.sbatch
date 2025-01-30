#!/bin/bash
#SBATCH --job-name=0BC_UA0.02
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --mem-per-cpu=200G
#SBATCH -o ClusterMessages/out/myscript_0BC_UA0.02_%j.out
#SBATCH -e ClusterMessages/error/myscript_0BC_UA0.02_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 0BC_UA0.02 \
    --Data BreastCancer \
    --Seed 0 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.02 \
    --Type Classification \
    --Output BreastCancer/TreeFarms/Raw/0BC_UA0.02.pkl