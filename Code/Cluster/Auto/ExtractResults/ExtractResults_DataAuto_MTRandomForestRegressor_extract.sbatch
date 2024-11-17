#!/bin/bash
#SBATCH --job-name=ExtractResults_DataAuto_MTRandomForestRegressor_extract
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/extract_ExtractResults_DataAuto_MTRandomForestRegressor_%j.out
#SBATCH -e ClusterMessages/error/extract_ExtractResults_DataAuto_MTRandomForestRegressor_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/ExtractError.py \
    --DataType Auto \
    --SelectorType '["PassiveLearning", "GSx", "GSy", "iGS"]' \
    --ModelType RandomForestRegressor \