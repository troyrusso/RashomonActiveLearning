#!/bin/bash

cd RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "Auto" \
    --ModelType "TreeFarms" \
    --Categories '["Auto_ENDING1.pkl", "Auto_ENDING2.pkl", "Auto_ENDING3.pkl", "Auto_ENDING4.pkl"]'
