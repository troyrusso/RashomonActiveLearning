#!/bin/bash
#SBATCH --job-name=ExtractResults_DataBostonHousing_MTRidgeRegression_extract
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/extract_ExtractResults_DataBostonHousing_MTRidgeRegression_%j.out
#SBATCH -e ClusterMessages/error/extract_ExtractResults_DataBostonHousing_MTRidgeRegression_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/ExtractError.py \
    --DataType BostonHousing \
    --SelectorType '["PassiveLearning", "GSx", "GSy", "iGS"]' \
    --ModelType RidgeRegression \