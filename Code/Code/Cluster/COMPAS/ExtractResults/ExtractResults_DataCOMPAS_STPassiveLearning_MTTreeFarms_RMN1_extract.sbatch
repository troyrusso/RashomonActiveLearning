#!/bin/bash
#SBATCH --job-name=ExtractResults_DataCOMPAS_STPassiveLearning_MTTreeFarms_RMN1_extract
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/extract_ExtractResults_DataCOMPAS_STPassiveLearning_MTTreeFarms_RMN1_%j.out
#SBATCH -e ClusterMessages/error/extract_ExtractResults_DataCOMPAS_STPassiveLearning_MTTreeFarms_RMN1_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/ExtractError.py \
    --DataType COMPAS \
    --SelectorType 'PassiveLearning' \
    --ModelType TreeFarmsRashomonNum1 \