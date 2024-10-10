### Summary:
### Inputs:
### Output:

### Set Up ###
library(tidyverse)
rm(list=ls())
dir = paste0("/Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/")
ExpandGridCombinations = expand.grid(RashomonModelNumLimit = c(10,25,100))

### Run SLURM Simulations ###
Simulation_Combinations = ExpandGridCombinations %>%
  mutate(JobName = paste0("Extract", "_",
                          RashomonModelNumLimit),
         Output = paste0("Results/Extracted",
                         RashomonModelNumLimit,
                         ".RData"))

# Loop through each row
for (i in 1:nrow(Simulation_Combinations)) {
  RashomonModelNumLimit = Simulation_Combinations[i, "RashomonModelNumLimit"]
  JobName = Simulation_Combinations[i, "JobName"]
  output = Simulation_Combinations[i, "Output"]

  # Create .sbatch file for the current simulation
  sbatch_file <- file(paste0(dir,
                             "/ExtractResults/",
                             JobName, 
                             ".sbatch"), "w")
  writeLines(
    c(
      "#!/bin/bash",
      paste("#SBATCH --job-name", JobName),
      "#SBATCH --partition short",
      "#SBATCH --ntasks 1",
      "#SBATCH --time 11:59:00",
      "#SBATCH --mem-per-cpu=2000",
      paste("#SBATCH -o ClusterMessages/out/myscript_", 
            JobName, 
            "_%j.out", 
            sep=""),
      paste("#SBATCH -e ClusterMessages/error/myscript_", 
            JobName, 
            "_%j.err", 
            sep=""),
      "#SBATCH --mail-type=ALL",
      "#SBATCH --mail-user=simondn@uw.edu",
      "",
      "cd ~/RashomonActiveLearning",
      "module load R/4.2.2-foss-2022b",
      "Rscript Code/functions/Auxiliary/Extract/ExtractErrorVec.R \\",
      
     paste("    --JobName ", JobName, "\\", sep=""),
     paste("    --RashomonModelNumLimit ", RashomonModelNumLimit, "\\", sep=""),
     paste("    --output ", output, "\\", sep="")
      
    ),
    con=sbatch_file
  )
  close(sbatch_file)
}

print("Sbatch files generated successfully.")


