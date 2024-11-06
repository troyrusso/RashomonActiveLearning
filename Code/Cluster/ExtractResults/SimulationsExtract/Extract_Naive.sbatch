#!/bin/bash
#SBATCH --job-name Extract_Naive
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Extract_Naive_%j.out
#SBATCH -e ClusterMessages/error/myscript_Extract_Naive_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Auxiliary/Extract/ExtractErrorVec_Simulation.R \
    --JobName Extract_Naive\
    --RashomonModelNumLimit NULL\
    --Method Factorial_BreakingTies\
    --output Results/Extracted/Simulations/Extracted_Naive.RData\
