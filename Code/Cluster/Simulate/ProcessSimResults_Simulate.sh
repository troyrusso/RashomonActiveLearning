#!/bin/bash

cd RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "Simulate" \
    --ModelType "TreeFarms" \
    --Categories '["PassiveLearning_MTTreeFarmsRashomonNum11.pkl", "RashomonQBC_MTTreeFarmsRashomonNum11.pkl", "RashomonQBC_MTTreeFarmsRashomonNum1010.pkl", "RashomonQBC_MTTreeFarmsRashomonNum100100.pkl"]'
