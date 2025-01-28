#!/bin/bash
#SBATCH --job-name=18CarEvaluation_FRT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o ClusterMessages/out/myscript_18CarEvaluation_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_18CarEvaluation_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 18CarEvaluation_FRT \
    --Data CarEvaluation \
    --Seed 18 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.04 \
    --Output 18CarEvaluation_FRT.pkl