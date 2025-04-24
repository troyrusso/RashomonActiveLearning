#!/bin/bash
#SBATCH --job-name=48BC_PLA0_D0B5
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH -o ClusterMessages/out/myscript_48BC_PLA0_D0B5_%j.out
#SBATCH -e ClusterMessages/error/myscript_48BC_PLA0_D0B5_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 48BC_PLA0_D0B5 \
    --Data BreastCancer \
    --Seed 48 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType PassiveLearning \
    --ModelType RandomForestClassificationFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.0 \
    --Type Classification \
    --BatchSize 5 \
    --DiversityWeight 0.0 \
    --Output BreastCancer/RandomForestClassification/Raw/48BC_PLA0_D0B5.pkl