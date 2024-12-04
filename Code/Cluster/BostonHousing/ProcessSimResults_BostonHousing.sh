#!/bin/bash

cd RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "BostonHousing" \
    --ModelType "TreeFarms" \
    --Categories '["BostonHousing_ENDING1.pkl", "BostonHousing_ENDING2.pkl", "BostonHousing_ENDING3.pkl", "BostonHousing_ENDING4.pkl"]'
