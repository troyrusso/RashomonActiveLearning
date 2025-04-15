#!/bin/bash
#
# RunResultsAnalysis.sh
# Script to run the Rashomon Set Active Learning analysis for a single dataset and value
#
# Usage: ./RunResultsAnalysis.sh DataType Value
# Example: ./RunResultsAnalysis.sh BankNote 025

# Check if both required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 DataType Value"
    echo "Example: $0 BankNote 025"
    exit 1
fi

# Get command line arguments
DATA_TYPE="$1"
VALUE="$2"

# Change to the script's directory
cd "$(dirname "$0")"

# Define paths
SCRIPT_DIR="$PWD"
PROJECT_ROOT="$(dirname "$(dirname "$PWD")")"  # Up two levels from Code/Analysis
RESULTS_DIR="$PROJECT_ROOT/Results"

# Make sure the Python script is executable
chmod +x RunResultsAnalysis.py

# Check if virtual environment exists and activate it if needed
VENV_PATH="$PROJECT_ROOT/venv"
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# Log file for tracking the analysis
LOG_FILE="$SCRIPT_DIR/analysis_${DATA_TYPE}_${VALUE}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"

echo "Starting Rashomon Set Active Learning analysis at $(date)" | tee -a "$LOG_FILE"
echo "Dataset: $DATA_TYPE" | tee -a "$LOG_FILE"
echo "Value: $VALUE" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"

# Check if Python modules are installed
echo "Checking dependencies..." | tee -a "$LOG_FILE"
MISSING_DEPS=0

for module in numpy pandas matplotlib scipy; do
    if ! python -c "import $module" &>/dev/null; then
        echo "Missing Python module: $module" | tee -a "$LOG_FILE"
        echo "Please install with: pip install $module" | tee -a "$LOG_FILE"
        MISSING_DEPS=1
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo "Missing dependencies detected. Please install them and retry." | tee -a "$LOG_FILE"
    exit 1
fi

# Run the analysis
echo "Running analysis for $DATA_TYPE with value $VALUE" | tee -a "$LOG_FILE"
python RunResultsAnalysis.py "$DATA_TYPE" "$VALUE" --BaseDirectory "$RESULTS_DIR/" 2>&1 | tee -a "$LOG_FILE"

# Check if the analysis was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Analysis completed successfully for $DATA_TYPE with value $VALUE" | tee -a "$LOG_FILE"
else
    echo "❌ Error in analysis for $DATA_TYPE with value $VALUE" | tee -a "$LOG_FILE"
    echo "Please check the log file for details: $LOG_FILE"
    exit 1
fi

echo "=====================================================" | tee -a "$LOG_FILE"
echo "Analysis completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"