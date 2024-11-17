#!/bin/bash
#SBATCH --job-name=MakeSbatchFiles_Simulate
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/MakeSbatchFiles_Simulate%j.out
#SBATCH -e ClusterMessages/error/MakeSbatchFiles_Simulate%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/CreateSimulationSbatch.py \
    --DataType Simulate \
