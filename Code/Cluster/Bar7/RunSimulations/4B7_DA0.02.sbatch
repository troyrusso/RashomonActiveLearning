#!/bin/bash
#SBATCH --job-name=4B7_DA0.02
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --ntasks=1
#SBATCH --time=2-11:59:00
#SBATCH --mem-per-cpu=100G
#SBATCH -o ClusterMessages/out/myscript_4B7_DA0.02_%j.out
#SBATCH -e ClusterMessages/error/myscript_4B7_DA0.02_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 4B7_DA0.02 \
    --Data Bar7 \
    --Seed 4 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.02 \
    --Type Classification \
    --Output Bar7/TreeFarms/Raw/4B7_DA0.02.pkl