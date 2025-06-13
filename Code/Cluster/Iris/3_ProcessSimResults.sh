#!/bin/bash

# This script is designed to be run from your dataset-specific cluster directory, e.g.,
# ~/RashomonActiveLearning/Code/Cluster/BankNote/
# It then navigates to the project root to run the Python script.

### Get Current Directory Name (e.g., BankNote) ###
CURRENT_DATASET=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DATASET"

# Navigate to the project root directory
# Adjust this path if your project root is not directly at ~/RashomonActiveLearning
cd "$HOME/RashomonActiveLearning" || { echo "Error: Could not navigate to project root."; exit 1; }

# Define the path to the Python aggregation script
PROCESS_SCRIPT="Code/utils/Auxiliary/ProcessSimulationResults.py"

echo "--- Extracting Results for $CURRENT_DATASET ---"

# 1. RF_PL: RandomForestClassifierPredictor + PassiveLearningSelector
#    JobName: 0BN_PL_RF_B<BatchSize>
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "RandomForestClassifierPredictor" \
    --Categories "_PL_RF_"

# 2. GPC_PL: GaussianProcessClassifierPredictor + PassiveLearningSelector
#    JobName: 0BN_PL_GPC_B<BatchSize>_KTRBF_KLS1.0_KNU1.5_K0.0 (etc.)
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "GaussianProcessClassifierPredictor" \
    --Categories "_PL_GPC_"

# 3. BNN_PL: BayesianNeuralNetworkPredictor + PassiveLearningSelector
#    JobName: 0BN_PL_BNN_B<BatchSize>_HS...
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "BayesianNeuralNetworkPredictor" \
    --Categories "_PL_BNN_"

# 4. BNN_BALD: BayesianNeuralNetworkPredictor + BALDSelector
#    JobName: 0BN_BALD_BNN_B<BatchSize>_HS...
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "BayesianNeuralNetworkPredictor" \
    --Categories "_BALD_BNN_"

# 5. GPC_BALD: GaussianProcessClassifierPredictor + BALDSelector
#    JobName: 0BN_BALD_GPC_B<BatchSize>_KTRBF...
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "GaussianProcessClassifierPredictor" \
    --Categories "_BALD_GPC_"

# # 6. UNREAL: TreeFarmsPredictor + BatchQBCSelector + UniqueErrorsInput=1
# #    JobName: 0BN_UNREAL_UEI1A<Threshold>_DW<Weight>_DEW<Weight>_B<BatchSize>
# python "$PROCESS_SCRIPT" \
#     --DataType "$CURRENT_DATASET" \
#     --ModelType "TreeFarmsPredictor" \
#     --Categories "_UNREAL_UEI1A" # Match start of category including UEI1A

# # 7. DUREAL: TreeFarmsPredictor + BatchQBCSelector + UniqueErrorsInput=0
# #    JobName: 0BN_DUREAL_UEI0A<Threshold>_DW<Weight>_DEW<Weight>_B<BatchSize>
# python "$PROCESS_SCRIPT" \
#     --DataType "$CURRENT_DATASET" \
#     --ModelType "TreeFarmsPredictor" \
    # --Categories "_DUREAL_UEI0A" # Match start of category including UEI0A

# 8. RF_QBC: RandomForestClassifierPredictor + BatchQBCSelector
#    JobName: 0BN_QBC_RF_UEI0A0_DW<Weight>_DEW<Weight>_B<BatchSize>
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "RandomForestClassifierPredictor" \
    --Categories "_QBC_RF_" # Match start of category including UEI0A

# 9. UNREAL_LFR: LFRPredictor + BatchQBCSelector + UniqueErrorsInput=1
#    JobName: 0BN_UNREAL_LFR_UEI1A<Threshold>_DW<Weight>_DEW<Weight>_B<BatchSize>
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "LFRPredictor" \
    --Categories "_UNREAL_LFR_" # Match start of category including UEI1A

# 10. DUREAL_LFR: LFRPredictor + BatchQBCSelector + UniqueErrorsInput=0
#     JobName: 0BN_DUREAL_LFR_UEI0A<Threshold>_DW<Weight>_DEW<Weight>_B<BatchSize>
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "LFRPredictor" \
    --Categories "_DUREAL_LFR_" # Match start of category including UEI0A

echo "--- All Extraction Commands Submitted ---"