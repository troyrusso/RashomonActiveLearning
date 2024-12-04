#!/bin/bash

# cd RashomonActiveLearning
# module load python
# python Code/utils/Auxiliary/ProcessSimulationResults.py \
#     --DataType "BostonHousingBinned" \
#     --ModelType "TreeFarms" \
#     --Categories '["PassiveLearning_MTTreeFarmsRashomonNum11.pkl", "RashomonQBC_MTTreeFarmsRashomonNum11.pkl", "RashomonQBC_MTTreeFarmsRashomonNum1010.pkl", "RashomonQBC_MTTreeFarmsRashomonNum100100.pkl"]'

cd RashomonActiveLearning
module load python
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "BostonHousingBinned" \
    --ModelType "RandomForestClassification" \
    --Categories '["STTreeEnsembleQBC_MTRandomForestClassificationRashomonNum100.pkl"]'