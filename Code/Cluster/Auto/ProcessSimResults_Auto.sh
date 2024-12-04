#!/bin/bash

cd RashomonActiveLearning
module load Python
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "Auto" \
    --ModelType "TreeFarms" \
    --Categories '["Auto_ENDING1.pkl", "Auto_ENDING2.pkl", "Auto_ENDING3.pkl", "Auto_ENDING4.pkl"]'
