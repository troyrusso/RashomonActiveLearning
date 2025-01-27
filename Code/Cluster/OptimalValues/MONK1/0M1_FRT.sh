#!/bin/bash
#SBATCH --job-name=FindThreshold_BC
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o ClusterMessages/out/FindThreshold_BC.06_%j.out
#SBATCH -e ClusterMessages/error/FindThreshold_BC.06_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName OptimalThresholdTest \
    --Data BreastCancer \
    --Seed 0 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.1 \
    --Output TEST.pkl



