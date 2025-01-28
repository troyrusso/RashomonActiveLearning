#!/bin/bash
#SBATCH --job-name=3BreastCancer_FRT
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o ClusterMessages/out/myscript_3BreastCancer_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_3BreastCancer_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 3BreastCancer_FRT \
    --Data BreastCancer \
    --Seed 3 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.1 \
    --Output 3BreastCancer_FRT.pkl