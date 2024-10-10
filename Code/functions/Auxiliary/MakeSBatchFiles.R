### Summary:
### Inputs:
### Output:

# Read CSV file
rm(list=ls())
dir = paste0("/Users/simondn/Documents/RashomonActiveLearning/Code/Cluster/")
ParameterVector <- read.csv(paste0(dir,"/ParameterVectorALL.csv"))

# Loop through each row
for (i in 1:nrow(ParameterVector)) {
  job_name = ParameterVector[i, "JobName"]
  seed = ParameterVector[i, "seed"]
  ModelType = ParameterVector[i, "ModelType"]
  SelectorType = ParameterVector[i, "SelectorType"]
  N = ParameterVector[i, "N"]
  K = ParameterVector[i, "K"]
  NClass = ParameterVector[i, "NClass"]
  CovCorrVal = ParameterVector[i, "CovCorrVal"]
  NBins = ParameterVector[i, "NBins"]
  TestProportion = ParameterVector[i, "TestProportion"]
  SelectorN = ParameterVector[i, "SelectorN"]
  InitialN = ParameterVector[i, "InitialN"]
  reg = ParameterVector[i, "reg"]
  theta = ParameterVector[i, "theta"]
  RashomonModelNumLimit = ParameterVector[i, "RashomonModelNumLimit"]
  LabelName = ParameterVector[i, "LabelName"]
  output = ParameterVector[i, "Output"]
  

  
  # Create .sbatch file for the current simulation
  sbatch_file <- file(paste0(dir,
                             "/RunSimulation/",
                             job_name, 
                             ".sbatch"), "w")
  writeLines(
    c(
      "#!/bin/bash",
      paste("#SBATCH --job-name", job_name),
      "#SBATCH --partition short",
      "#SBATCH --ntasks 1",
      "#SBATCH --time 11:59:00",
      "#SBATCH --mem-per-cpu= 1000",
      paste("#SBATCH -o ClusterMessages/out/myscript_", 
            job_name, 
            "_%j.out", 
            sep=""),
      paste("#SBATCH -e ClusterMessages/error/myscript_", 
            job_name, 
            "_%j.err", 
            sep=""),
      "#SBATCH --mail-type=ALL",
      "#SBATCH --mail-user=simondn@uw.edu",
      "",
      "cd ~/RashomonActiveLearning",
      "module load R/4.2.2-foss-2022b",
      "Rscript Code/functions/Main/RunSimulation.R \\",
      
     paste("    --job_name ", job_name, "\\", sep=""),
     paste("    --seed ", seed, "\\", sep=""),
     paste("    --ModelType ", ModelType, "\\", sep=""),
     paste("    --SelectorType ", SelectorType, "\\", sep=""),
     paste("    --N ", N, "\\", sep=""),
     paste("    --K ", K, "\\", sep=""),
     paste("    --NClass ", NClass, "\\", sep=""),
     paste("    --CovCorrVal ", CovCorrVal, "\\", sep=""),
     paste("    --NBins ", NBins, "\\", sep=""),
     paste("    --TestProportion ", TestProportion, "\\", sep=""),
     paste("    --SelectorN ", SelectorN, "\\", sep=""),
     paste("    --InitialN ", InitialN, "\\", sep=""),
     paste("    --reg ", reg, "\\", sep=""),
     paste("    --theta ", theta, "\\", sep=""),
     paste("    --RashomonModelNumLimit ", RashomonModelNumLimit, "\\", sep=""),
     paste("    --LabelName ", LabelName, "\\", sep=""),
     paste("    --output ", output, "\\", sep="")
      
    ),
    con=sbatch_file
  )
  close(sbatch_file)
}

print("Sbatch files generated successfully.")


