#!/bin/bash
#SBATCH --job-name Sim_50_Linear_Random_500_3_0_4_0.2_1_10_0.1_2_0_Y
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 11:59:00
#SBATCH --mem-per-cpu=30000
#SBATCH -o ClusterMessages/out/myscript_Sim_50_Linear_Random_500_3_0_4_0.2_1_10_0.1_2_0_Y_%j.out
#SBATCH -e ClusterMessages/error/myscript_Sim_50_Linear_Random_500_3_0_4_0.2_1_10_0.1_2_0_Y_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load R/4.2.2-foss-2022b
Rscript Code/functions/Main/RunSimulation.R \
    --job_name Sim_50_Linear_Random_500_3_0_4_0.2_1_10_0.1_2_0_Y\
    --seed 50\
    --ModelType Linear\
    --SelectorType Random\
    --N 500\
    --K 3\
    --CovCorrVal 0\
    --NBins 4\
    --TestProportion 0.2\
    --SelectorN 1\
    --InitialN 10\
    --reg 0.1\
    --theta 2\
    --RashomonModelNumLimit 0\
    --LabelName Y\
    --output Results/SimulationRaw/50_Linear_Random_500_3_0_4_0.2_1_10_0.1_2_0_.RData\
