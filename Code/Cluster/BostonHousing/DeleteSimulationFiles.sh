#!/bin/bash

# Navigate to the RunSimulations directory
cd RunSimulations

# Run delete_sbatch.sh
if [[ -f delete_sbatch.sh ]]; then
    echo "Running delete_sbatch.sh..."
    bash delete_sbatch.sh
else
    echo "delete_sbatch.sh not found in RunSimulations."
fi

# Navigate to ClusterMessages/out and run delete_out.sh
cd ClusterMessages/out
if [[ -f delete_out.sh ]]; then
    echo "Running delete_out.sh..."
    bash delete_out.sh
else
    echo "delete_out.sh not found in ClusterMessages/out."
fi

# Navigate to ClusterMessages/error and run delete_error.sh
cd ../error
if [[ -f delete_error.sh ]]; then
    echo "Running delete_error.sh..."
    bash delete_error.sh
else
    echo "delete_error.sh not found in ClusterMessages/error."
fi

echo "All delete scripts executed."