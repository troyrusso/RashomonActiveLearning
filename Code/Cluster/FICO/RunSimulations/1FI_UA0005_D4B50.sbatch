#!/bin/bash
#SBATCH --job-name=1FI_UA0005_D4B50
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_1FI_UA0005_D4B50_%j.out
#SBATCH -e ClusterMessages/error/myscript_1FI_UA0005_D4B50_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/RunSimulation.py \
    --JobName 1FI_UA0005_D4B50 \
    --Data FICO \
    --Seed 1 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --SelectorType BatchQBCDiversityFunction \
    --ModelType TreeFarmsFunction \
    --UniqueErrorsInput 1 \
    --n_estimators 100 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.0005 \
    --Type Classification \
    --BatchSize 50 \
    --DiversityWeight 0.4 \
    --Output FICO/TreeFarms/Raw/1FI_UA0005_D4B50.pkl