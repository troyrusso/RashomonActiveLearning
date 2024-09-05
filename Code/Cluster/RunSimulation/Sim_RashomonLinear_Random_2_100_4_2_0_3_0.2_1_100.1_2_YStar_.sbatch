#!/bin/bash
#SBATCH --job-name Sim_RashomonLinear_Random_2_100_4_2_0_3_0.2_1_100.1_2_YStar_
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=500
#SBATCH -o ClusterMessages/out/myscript_Sim_RashomonLinear_Random_2_100_4_2_0_3_0.2_1_100.1_2_YStar__%j.out
#SBATCH -e ClusterMessages/error/myscript_Sim_RashomonLinear_Random_2_100_4_2_0_3_0.2_1_100.1_2_YStar__%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Main/RunSimulation.R \
    --job_name Sim_RashomonLinear_Random_2_100_4_2_0_3_0.2_1_100.1_2_YStar_\
    --seed 2\
    --ModelType RashomonLinear\
    --SelectorType Random\
    --N 100\
    --K 4\
    --NClass 2\
    --CovCorrVal 0\
    --NBins 3\
    --TestProportion 0.2\
    --SelectorN 1\
    --InitialN 10\
    --reg 0.1\
    --theta 2\
    --LabelName YStar\
    --output Results/2_RashomonLinear_Random_100_4_2_0_3_0.2_1_10.rds\
