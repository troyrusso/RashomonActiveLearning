#!/bin/bash
#SBATCH --job-name Extract_Random
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Extract_Random_%j.out
#SBATCH -e ClusterMessages/error/myscript_Extract_Random_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Auxiliary/Extract/ExtractErrorVec_Vanilla.R \
    --JobName Extract_Random\
    --Method Random\
    --N 500\
    --K 6\
    --output Results/Extracted/Simulations/VanillaRandom6.RData\
