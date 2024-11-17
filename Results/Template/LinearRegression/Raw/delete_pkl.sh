#!/bin/bash

# Function to delete all sbatch files
delete_pkl_files() {
    if ls *.pkl 1> /dev/null 2>&1; then
        echo "Deleting all .pkl files in $(pwd)..."
        rm *.pkl
        echo "All .pkl files deleted."
    else
        echo "No .pkl files found in $(pwd)."
    fi
}

# Execute the function when the script is run
delete_sbatch_files
