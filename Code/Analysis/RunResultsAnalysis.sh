#!/bin/bash
#
# run_analysis.sh
# Script to run the Rashomon Set Active Learning analysis 
# for multiple datasets and values
#

# Change to the correct working directory
cd "$(dirname "$0")"

# Define paths
SCRIPT_DIR="$PWD"
PROJECT_ROOT="$(dirname "$(dirname "$PWD")")"  # Up two levels from Code/Analysis
RESULTS_DIR="$PROJECT_ROOT/Results"

# Make sure the Python script is executable
chmod +x data_analysis.py

# Define datasets to analyze - use the actual dataset directory names from your tree
DATASETS=(
    "BankNote"
    "Bar7"
    "BreastCancer"
    "CarEvaluation"
    "COMPAS"
    "FICO"
    "Haberman"
    "Iris"
    "MONK1"
    "MONK3"
)

# Define values to analyze
VALUES=(
    "005"
    "010"
    "015"
    "020"
    "025"
)

# Log file for tracking the analysis
LOG_FILE="$SCRIPT_DIR/analysis_log_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"

echo "Starting Rashomon Set Active Learning analysis at $(date)" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"

# Run analysis for each dataset and value combination
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset" | tee -a "$LOG_FILE"
    echo "-----------------------------------" | tee -a "$LOG_FILE"
    
    for value in "${VALUES[@]}"; do
        echo "Running analysis for $dataset with value $value" | tee -a "$LOG_FILE"
        
        # Run the Python script with the current dataset and value
        python data_analysis.py "$dataset" "$value" --BaseDirectory "$RESULTS_DIR/" 2>&1 | tee -a "$LOG_FILE"
        
        # Check if the analysis was successful
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ Analysis completed successfully for $dataset with value $value" | tee -a "$LOG_FILE"
        else
            echo "❌ Error in analysis for $dataset with value $value" | tee -a "$LOG_FILE"
        fi
        
        echo "-----------------------------------" | tee -a "$LOG_FILE"
    done
    
    echo "" | tee -a "$LOG_FILE"
done

echo "=====================================================" | tee -a "$LOG_FILE"
echo "All analyses completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
