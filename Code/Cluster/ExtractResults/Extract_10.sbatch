#!/bin/bash
#SBATCH --job-name Extract_10
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Extract_10_%j.out
#SBATCH -e ClusterMessages/error/myscript_Extract_10_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Auxiliary/Extract/ExtractErrorVec.R \
    --JobName Extract_10\
    --RashomonModelNumLimit 10\
    --output Results/Extracted10.RData\
