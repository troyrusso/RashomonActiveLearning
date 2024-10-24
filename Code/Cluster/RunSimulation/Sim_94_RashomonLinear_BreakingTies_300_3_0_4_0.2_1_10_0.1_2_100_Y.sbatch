#!/bin/bash
#SBATCH --job-name Sim_94_RashomonLinear_BreakingTies_300_3_0_4_0.2_1_10_0.1_2_100_Y
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Sim_94_RashomonLinear_BreakingTies_300_3_0_4_0.2_1_10_0.1_2_100_Y_%j.out
#SBATCH -e ClusterMessages/error/myscript_Sim_94_RashomonLinear_BreakingTies_300_3_0_4_0.2_1_10_0.1_2_100_Y_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Main/RunSimulation.R \
    --job_name Sim_94_RashomonLinear_BreakingTies_300_3_0_4_0.2_1_10_0.1_2_100_Y\
    --seed 94\
    --ModelType RashomonLinear\
    --SelectorType BreakingTies\
    --N 300\
    --K 3\
    --CovCorrVal 0\
    --NBins 4\
    --TestProportion 0.2\
    --SelectorN 1\
    --InitialN 10\
    --reg 0.1\
    --theta 2\
    --RashomonModelNumLimit 100\
    --LabelName Y\
    --output Results/SimulationRaw/94_RashomonLinear_BreakingTies_300_3_0_4_0.2_1_10_0.1_2_100_.RData\
