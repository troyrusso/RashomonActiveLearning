#!/bin/bash

# Function to delete all err files
delete_results_files() {
    echo "Deleting all .pkl files in $(pwd)..."
    rm *.pkl
}

# Execute the function when the script is run
delete_results_files
