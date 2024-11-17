#!/bin/bash
#SBATCH --job-name=CreateExtractFiles_Template
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/CreateExtractFiles_Template%j.out
#SBATCH -e ClusterMessages/error/CreateExtractFiles_Template%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/CreateExtractSbatch.py \
    --DataType Template \
