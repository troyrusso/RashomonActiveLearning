#!/bin/bash
#SBATCH --job-name=96CarEvaluation_FRT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o ClusterMessages/out/myscript_96CarEvaluation_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_96CarEvaluation_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 96CarEvaluation_FRT \
    --Data CarEvaluation \
    --Seed 96 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.1 \
    --Output 96CarEvaluation_FRT.pkl