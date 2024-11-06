#!/bin/bash
#SBATCH --job-name Sim_41_Linear_BreakingTies_500_6_0_4_0.2_1_10_0.1_2_10_Y
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Sim_41_Linear_BreakingTies_500_6_0_4_0.2_1_10_0.1_2_10_Y_%j.out
#SBATCH -e ClusterMessages/error/myscript_Sim_41_Linear_BreakingTies_500_6_0_4_0.2_1_10_0.1_2_10_Y_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Main/RunSimulation.R \
    --job_name Sim_41_Linear_BreakingTies_500_6_0_4_0.2_1_10_0.1_2_10_Y\
    --seed 41\
    --ModelType Linear\
    --SelectorType BreakingTies\
    --N 500\
    --K 6\
    --CovCorrVal 0\
    --NBins 4\
    --TestProportion 0.2\
    --SelectorN 1\
    --InitialN 10\
    --reg 0.1\
    --theta 2\
    --RashomonModelNumLimit 10\
    --LabelName Y\
    --output Results/Vanilla/41_Linear_BreakingTies_500_6_0_4_0.2_1_10_0.1_2_10_.RData\
