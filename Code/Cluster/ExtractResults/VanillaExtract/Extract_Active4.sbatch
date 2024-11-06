#!/bin/bash
#SBATCH --job-name Extract_BT
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Extract_BT_%j.out
#SBATCH -e ClusterMessages/error/myscript_Extract_BT_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Auxiliary/Extract/ExtractErrorVec_Vanilla.R \
    --JobName Extract_BT\
    --K 4\
    --Method BreakingTies\
    --output Results/Extracted/Simulations/Extracted_Naive.RData\
