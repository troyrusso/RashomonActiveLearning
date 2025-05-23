#!/bin/bash

### Get the current directory name ###
CURRENT_DIR=$(basename "$PWD")
echo "Current directory is: $CURRENT_DIR"

### Delete all Unprocessed Results files ##

# Check if the Random Forest Results directory exists
RF_DIR="../../../Results/$CURRENT_DIR/RandomForestClassification/Raw"
if [ -d "$RF_DIR" ]; then
    cd "$RF_DIR"
    
    # Remove Random Forest Results #
    if [ -f delete_results.sh ]; then
        bash delete_results.sh
    else
        # Check if there are any .pkl files before trying to delete
        if ls *.pkl 1> /dev/null 2>&1; then
            rm *.pkl
            echo "All .pkl results files in RandomForests deleted."
        else
            echo "No .pkl files found in RandomForests directory."
        fi
    fi
else
    echo "RandomForestClassification directory not found at expected path."
    exit 1
fi

# Check if the TreeFarms directory exists
TF_DIR="../../TreeFarms/Raw/"
if [ -d "$TF_DIR" ]; then
    cd "$TF_DIR"
    
    # Remove TreeFarms Results #
    if [ -f delete_results.sh ]; then
        bash delete_results.sh
    else
        # Check if there are any .pkl files before trying to delete
        if ls *.pkl 1> /dev/null 2>&1; then
            rm *.pkl
            echo "All .pkl results files in TreeFarms deleted."
        else
            echo "No .pkl files found in TreeFarms directory."
        fi
    fi
else
    echo "TreeFarms directory not found at expected path."
fi

# Check if the BayesianNeuralNetwork directory exists
TF_DIR="../../BayesianNeuralNetwork/Raw/"
if [ -d "$TF_DIR" ]; then
    cd "$TF_DIR"
    
    # Remove BayesianNeuralNetwork Results #
    if [ -f delete_results.sh ]; then
        bash delete_results.sh
    else
        # Check if there are any .pkl files before trying to delete
        if ls *.pkl 1> /dev/null 2>&1; then
            rm *.pkl
            echo "All .pkl results files in BayesianNeuralNetwork deleted."
        else
            echo "No .pkl files found in BayesianNeuralNetwork directory."
        fi
    fi
else
    echo "BayesianNeuralNetwork directory not found at expected path."
fi

echo "Raw results cleanup completed."