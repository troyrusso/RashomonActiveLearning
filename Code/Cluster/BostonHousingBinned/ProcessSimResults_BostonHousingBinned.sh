#!/bin/bash

# cd RashomonActiveLearning
# python Code/utils/Auxiliary/ProcessSimulationResults.py \
#     --DataType "BostonHousingBinned" \
#     --ModelType "TreeFarms" \
#     --Categories '["PassiveLearning_MTTreeFarmsRashomonNum11.pkl", "RashomonQBC_MTTreeFarmsRashomonNum11.pkl", "RashomonQBC_MTTreeFarmsRashomonNum1010.pkl", "RashomonQBC_MTTreeFarmsRashomonNum100100.pkl"]'

cd ~/RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "BostonHousingBinned" \
    --ModelType "TreeFarms" \
    --Categories '["8_STTreeEnsembleQBC_MTTreeFarms_UEI0_RashomonNum100.pkl",
                   "8_STTreeEnsembleQBC_MTTreeFarms_UEI0_RashomonNum10.pkl",
                   "8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum100.pkl",
                   "8_STTreeEnsembleQBC_MTTreeFarms_UEI1_RashomonNum10.pkl"]'
                   
cd ~/RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "BostonHousingBinned" \
    --ModelType "RandomForestClassification" \
    --Categories '["MTRandomForestClassification_UEI1_RashomonNum10.pkl"]'