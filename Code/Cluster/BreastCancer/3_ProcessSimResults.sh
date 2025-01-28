#!/bin/bash

### Current Directory Name ###
CURRENT_DIR=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DIR"
cd ~/RashomonActiveLearning

### Extract Random Forests Results ###
# python Code/utils/Auxiliary/ProcessSimulationResults.py \
#     --DataType "$CURRENT_DIR" \
#     --ModelType "RandomForestClassification" \
#     --Categories "RF0.0.pkl"

### Extract Duplicate TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "DA0.02.pkl"

### Extract Unique TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "UA0.02.pkl"
