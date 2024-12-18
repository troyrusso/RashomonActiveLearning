#!/bin/bash

### Extract Random Forests Results ###
cd ~/RashomonActiveLearning
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "Template" \
    --ModelType "RandomForestClassification" \
    --Categories '["MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01.pkl"]'

### Extract Duplicate TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "Template" \
    --ModelType "TreeFarms" \
    --Categories '["MTTreeFarms_UEI0_NE100_Reg0.01_RBA0.01.pkl"]'

### Extract Unique TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "Template" \
    --ModelType "TreeFarms" \
    --Categories '["MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01.pkl"]'