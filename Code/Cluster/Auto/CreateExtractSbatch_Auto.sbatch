#!/bin/bash
#SBATCH --job-name=MakeSbatchFiles_Auto
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/MakeSbatchFiles_Auto%j.out
#SBATCH -e ClusterMessages/error/MakeSbatchFiles_Auto%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/CreateSimulationSbatch.py \
    --DataType Auto \
