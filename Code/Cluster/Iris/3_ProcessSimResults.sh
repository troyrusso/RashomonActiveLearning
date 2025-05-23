#!/bin/bash

### Current Directory Name ###
CURRENT_DIR=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DIR"
cd ~/RashomonActiveLearning

### Extract Passive Learning Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "RandomForestClassification" \
    --Categories "PL_B"

### Extract Random Forests Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "RandomForestClassification" \
    --Categories "RF_"

### Extract BALD Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "BayesianNeuralNetwork" \
    --Categories "BALD_B"

### Extract Duplicate TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "DA025"

### Extract Unique TREEFARMS Results ###
python Code/utils/Auxiliary/ProcessSimulationResults.py \
    --DataType "$CURRENT_DIR" \
    --ModelType "TreeFarms" \
    --Categories "UA025"
