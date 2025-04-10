#!/bin/bash
#SBATCH --job-name=1FI_RFA0.0
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH -o ClusterMessages/out/myscript_1FI_RFA0.0_%j.out
#SBATCH -e ClusterMessages/error/myscript_1FI_RFA0.0_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 1FI_RFA0.0 \
    --Data FICO \
    --Seed 1 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType TreeEnsembleQBCFunction \
    --ModelType RandomForestClassificationFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.0 \
    --Type Classification \
    --Output FICO/RandomForestClassification/Raw/1FI_RFA0.0.pkl