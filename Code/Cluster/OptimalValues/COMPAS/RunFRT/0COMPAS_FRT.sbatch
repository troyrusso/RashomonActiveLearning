#!/bin/bash
#SBATCH --job-name=0COMPAS_FRT
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o ClusterMessages/out/myscript_0COMPAS_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_0COMPAS_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 0COMPAS_FRT \
    --Data COMPAS \
    --Seed 0 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.03 \
    --Output 0COMPAS_FRT.pkl