### Summary:
### Inputs:
### Output:

if(exists("directory")){
  
  source(paste0(directory,"Code/functions/Selector/BreakingTiesSelector.R"))
  source(paste0(directory,"Code/functions/Prediction/ModelTypeSwitch.R"))
  source(paste0(directory,"Code/functions/Selector/MostUncertainObservation.R"))
  source(paste0(directory,"Code/functions/Selector/RandomSelector.R"))
  source(paste0(directory,"Code/functions/Selector/RandomStart.R"))
  source(paste0(directory,"Code/functions/Selector/SelectorTypeSwitch.R"))
  source(paste0(directory,"Code/functions/Selector/StoppingCriteria.R"))
  source(paste0(directory,"Code/functions/Prediction/TestError.R"))
  source(paste0(directory,"Code/functions/Auxiliary/Validation.R"))
  
  source(paste0(directory,"Code/functions/Plot/ActiveLearningPlot.R"))
  source(paste0(directory,"Code/functions/Plot/ClassErrorPlot.R"))
  source(paste0(directory,"Code/functions/Plot/SelectorTypeComparisonPlotFuncDynamic.R"))

  source(paste0(directory,"Code/functions/Main/GenerateData.R"))
  source(paste0(directory,"Code/functions/Prediction/RashomonFunc.R"))
  source(paste0(directory,"Code/functions/Main/SimulationFunc.R"))
  
  # source(paste0(directory,"rashomon-tva-R-main/R/aggregate.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/count.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/find_rashomon_set.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/globals.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/loss.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/predictions.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/RashomonSet.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/rashomontva-package.R"))
  # source(paste0(directory,"rashomon-tva-R-main/R/utils.R"))
}else if(!exists("directory")){
  
  source("Code/functions/Selector/BreakingTiesSelector.R")
  source("Code/functions/Prediction/ModelTypeSwitch.R")
  source("Code/functions/Selector/MostUncertainObservation.R")
  source("Code/functions/Selector/RandomSelector.R")
  source("Code/functions/Selector/RandomStart.R")
  source("Code/functions/Selector/SelectorTypeSwitch.R")
  source("Code/functions/Selector/StoppingCriteria.R")
  source("Code/functions/Prediction/TestError.R")
  source("Code/functions/Auxiliary/Validation.R")
  
  source("Code/functions/Plot/ActiveLearningPlot.R")
  source("Code/functions/Plot/ClassErrorPlot.R")
  source("Code/functions/Plot/SelectorTypeComparisonPlotFuncDynamic.R")
  
  source("Code/functions/Main/GenerateData.R")
  source("Code/functions/Prediction/RashomonFunc.R")
  source("Code/functions/Main/SimulationFunc.R")

  # source("rashomon-tva-R-main/R/aggregate.R")
  # source("rashomon-tva-R-main/R/count.R")
  # source("rashomon-tva-R-main/R/find_rashomon_set.R")
  # source("rashomon-tva-R-main/R/globals.R")
  # source("rashomon-tva-R-main/R/loss.R")
  # source("rashomon-tva-R-main/R/predictions.R")
  # source("rashomon-tva-R-main/R/RashomonSet.R")
  # source("rashomon-tva-R-main/R/rashomontva-package.R")
  # source("rashomon-tva-R-main/R/utils.R")
}

