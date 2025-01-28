#!/bin/bash
#SBATCH --job-name=43MONK3_FRT
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o ClusterMessages/out/myscript_43MONK3_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_43MONK3_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 43MONK3_FRT \
    --Data MONK3 \
    --Seed 43 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.03 \
    --Output 43MONK3_FRT.pkl