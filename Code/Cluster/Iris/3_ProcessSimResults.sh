#!/bin/bash

### Current Directory Name ###
CURRENT_DIR=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DIR"
cd ~/RashomonActiveLearning

### Extract Passive Learning Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "RandomForestClassification" \
    --Categories "PLA0"

### Extract Random Forests Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "RandomForestClassification" \
    --Categories "RFA0"

### Extract Duplicate TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "DA01"

### Extract Unique TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "UA01"
