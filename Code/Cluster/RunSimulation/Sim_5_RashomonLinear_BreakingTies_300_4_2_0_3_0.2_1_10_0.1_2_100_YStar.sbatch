#!/bin/bash
#SBATCH --job-name Sim_5_RashomonLinear_BreakingTies_300_4_2_0_3_0.2_1_10_0.1_2_100_YStar
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=2000
#SBATCH -o ClusterMessages/out/myscript_Sim_5_RashomonLinear_BreakingTies_300_4_2_0_3_0.2_1_10_0.1_2_100_YStar_%j.out
#SBATCH -e ClusterMessages/error/myscript_Sim_5_RashomonLinear_BreakingTies_300_4_2_0_3_0.2_1_10_0.1_2_100_YStar_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Main/RunSimulation.R \
    --job_name Sim_5_RashomonLinear_BreakingTies_300_4_2_0_3_0.2_1_10_0.1_2_100_YStar\
    --seed 5\
    --ModelType RashomonLinear\
    --SelectorType BreakingTies\
    --N 300\
    --K 4\
    --NClass 2\
    --CovCorrVal 0\
    --NBins 3\
    --TestProportion 0.2\
    --SelectorN 1\
    --InitialN 10\
    --reg 0.1\
    --theta 2\
    --RashomonModelNumLimit 100\
    --LabelName YStar\
    --output Results/EmpiricalRaw/5_RashomonLinear_BreakingTies_300_4_2_0_3_0.2_1_10_0.1_2_100_.RData\
