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

# HPC environment setup - try to load Python module
if command -v module &> /dev/null; then
    echo "HPC environment detected, loading Python module..."
    # Try different module names common on HPC systems
    if module avail Python &> /dev/null; then
        module load Python
        echo "Loaded Python module"
    elif module avail python &> /dev/null; then
        module load python
        echo "Loaded python module"
    else
        echo "No Python module found, continuing with system Python"
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Using $PYTHON_VERSION"

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
        
        # Attempt to install missing module for user
        echo "Attempting to install $module..." | tee -a "$LOG_FILE"
        python -m pip install --user $module
        
        # Check if installation was successful
        if ! python -c "import $module" &>/dev/null; then
            echo "Failed to install $module automatically" | tee -a "$LOG_FILE"
            echo "Please install with: python -m pip install --user $module" | tee -a "$LOG_FILE"
            MISSING_DEPS=1
        else
            echo "Successfully installed $module" | tee -a "$LOG_FILE"
        fi
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo "Missing dependencies detected. Please install them and retry." | tee -a "$LOG_FILE"
    echo "For HPC environments, you may need to use:" | tee -a "$LOG_FILE"
    echo "  srun --pty --time=30 --mem-per-cpu=1000 --partition=build /bin/bash" | tee -a "$LOG_FILE"
    echo "  module load Python" | tee -a "$LOG_FILE"
    echo "  python -m pip install --user numpy pandas matplotlib scipy" | tee -a "$LOG_FILE"
    exit 1
fi

# Run the analysis
echo "Running analysis for $DATA_TYPE with value $VALUE" | tee -a "$LOG_FILE"

# For HPC environments, we might need to set the display variable for matplotlib
# (even if we're just saving to file and not displaying)
export MPLBACKEND=Agg

# Try to run with explicitly setting PYTHONPATH to include project root
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python RunResultsAnalysis.py "$DATA_TYPE" "$VALUE" --BaseDirectory "$RESULTS_DIR/" 2>&1 | tee -a "$LOG_FILE"

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