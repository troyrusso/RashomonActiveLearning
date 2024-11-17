#!/bin/bash
#SBATCH --job-name=CreateRunSimFiles_CPS
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/CreateRunSimFiles_CPS%j.out
#SBATCH -e ClusterMessages/error/CreateRunSimFiles_CPS%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/CreateRunSimSbatch.py \
    --DataType CPS \
