#!/bin/bash
#SBATCH --job-name=8BC_DA0.005
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_8BC_DA0.005_%j.out
#SBATCH -e ClusterMessages/error/myscript_8BC_DA0.005_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 8BC_DA0.005 \
    --Data BreastCancer \
    --Seed 8 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.005 \
    --Type Classification \
    --Output BreastCancer/TreeFarms/Raw/8BC_DA0.005.pkl