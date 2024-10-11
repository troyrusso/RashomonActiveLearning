#!/bin/bash
#SBATCH --job-name Sim_7_Factorial_Random_0.2_1_10_0.1_2_10_YStar
#SBATCH --partition medium
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Sim_7_Factorial_Random_0.2_1_10_0.1_2_10_YStar_%j.out
#SBATCH -e ClusterMessages/error/myscript_Sim_7_Factorial_Random_0.2_1_10_0.1_2_10_YStar_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Main/RunHousingSimulations.R \
    --job_name Sim_7_Factorial_Random_0.2_1_10_0.1_2_10_YStar\
    --seed 7\
    --ModelType Factorial\
    --SelectorType Random\
    --TestProportion 0.2\
    --SelectorN 1\
    --InitialN 10\
    --reg 0.1\
    --theta 2\
    --RashomonModelNumLimit 10\
    --output Results/SimulationRaw7_Factorial_Random_0.2_1_10_0.1_2_10_.RData\
