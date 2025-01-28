#!/bin/bash
#SBATCH --job-name=93CarEvaluation_FRT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o ClusterMessages/out/myscript_93CarEvaluation_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_93CarEvaluation_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 93CarEvaluation_FRT \
    --Data CarEvaluation \
    --Seed 93 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.08 \
    --Output 93CarEvaluation_FRT.pkl