#!/bin/bash
#SBATCH --job-name=49FI_PLA0_D0B50
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH -o ClusterMessages/out/myscript_49FI_PLA0_D0B50_%j.out
#SBATCH -e ClusterMessages/error/myscript_49FI_PLA0_D0B50_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 49FI_PLA0_D0B50 \
    --Data FICO \
    --Seed 49 \
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
    --BatchSize 50 \
    --DiversityWeight 0.0 \
    --Output FICO/RandomForestClassification/Raw/49FI_PLA0_D0B50.pkl