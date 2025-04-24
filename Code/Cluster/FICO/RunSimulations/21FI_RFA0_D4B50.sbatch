#!/bin/bash
#SBATCH --job-name=21FI_RFA0_D4B50
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH -o ClusterMessages/out/myscript_21FI_RFA0_D4B50_%j.out
#SBATCH -e ClusterMessages/error/myscript_21FI_RFA0_D4B50_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 21FI_RFA0_D4B50 \
    --Data FICO \
    --Seed 21 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType BatchQBCDiversityFunction \
    --ModelType RandomForestClassificationFunction \
    --UniqueErrorsInput 0 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.0 \
    --Type Classification \
    --BatchSize 50 \
    --DiversityWeight 0.4 \
    --Output FICO/RandomForestClassification/Raw/21FI_RFA0_D4B50.pkl