#!/bin/bash

### Get the current directory name ###
CURRENT_DIR=$(basename "$PWD")
echo "Current directory is: $CURRENT_DIR"

### Navigate to RunSimulations Directory ###
cd RunSimulations

### Delete all .sbatch files ###
bash delete_sbatch.sh
echo "All .sbatch files deleted."

### Delete all .out files ###
cd ClusterMessages/out
bash delete_out.sh
echo "All .out files deleted."

### Delete all .err files ###
cd ../error
bash delete_error.sh
echo "All .error files deleted."

### Delete all Unprocessed Results files ##
# Remove Random Forest Results #
cd ../../../../../../Results/"$CURRENT_DIR"/RandomForestClassification/Raw
rm *.pkl

# Remove TreeFarms Results #
cd ../../TreeFarms/Raw/
rm *.pkl
