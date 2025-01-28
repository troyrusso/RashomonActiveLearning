#!/bin/bash
#SBATCH --job-name=36MONK1_FRT
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o ClusterMessages/out/myscript_36MONK1_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_36MONK1_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 36MONK1_FRT \
    --Data MONK1 \
    --Seed 36 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.035 \
    --Output 36MONK1_FRT.pkl