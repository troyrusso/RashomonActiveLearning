#!/bin/bash

### Get the current directory name ###
CURRENT_DIR=$(basename "$PWD")
echo "Current directory is: $CURRENT_DIR"

### Navigate to RunSimulations Directory ###
cd RunSimulations

### Delete all .sbatch files ###
bash delete_sbatch.sh

### Delete all .out files ###
cd ClusterMessages/out
bash delete_out.sh

### Delete all .err files ###
cd ../error
bash delete_error.sh

### Delete all Unprocessed Results files ##
# Remove Random Forest Results #
cd ../../../../../../Results/"$CURRENT_DIR"/RandomForestClassification/Raw
bash delete_results.sh
echo "All .pkl results files in RandomForests deleted."

# Remove TreeFarms Results #
cd ../../TreeFarms/Raw/
bash delete_results.sh
echo "All .pkl results files in TreeFarms deleted."

