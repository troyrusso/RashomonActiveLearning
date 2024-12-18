#!/bin/bash

# Function to delete all err files
delete_error_files() {
    # Check for err files in the current directory
    if ls *.err 1> /dev/null 2>&1; then
        echo "Deleting all .err files in $(pwd)..."
        rm *.err
        echo "All .err files deleted."
    else
        echo "No .err files found in $(pwd)."
    fi
}

# Execute the function when the script is run
delete_error_files
