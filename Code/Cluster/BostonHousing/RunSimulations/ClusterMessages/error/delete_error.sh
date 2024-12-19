#!/bin/bash

# Function to delete all err files
delete_error_files() {
    echo "Deleting all .err files in $(pwd)..."
    rm *.err
}

# Execute the function when the script is run
delete_error_files
