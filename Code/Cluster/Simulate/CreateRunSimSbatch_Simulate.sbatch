#!/bin/bash
#SBATCH --job-name=CreateRunSimFiles_Simulate
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/CreateRunSimFiles_Simulate%j.out
#SBATCH -e ClusterMessages/error/CreateRunSimFiles_Simulate%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/CreateRunSimSbatch.py \
    --DataType Simulate \
