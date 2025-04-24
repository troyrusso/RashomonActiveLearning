#!/bin/bash
#SBATCH --job-name=23BC_UA075_D4B5
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_23BC_UA075_D4B5_%j.out
#SBATCH -e ClusterMessages/error/myscript_23BC_UA075_D4B5_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 23BC_UA075_D4B5 \
    --Data BreastCancer \
    --Seed 23 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType BatchQBCDiversityFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.075 \
    --Type Classification \
    --BatchSize 5 \
    --DiversityWeight 0.4 \
    --Output BreastCancer/TreeFarms/Raw/23BC_UA075_D4B5.pkl