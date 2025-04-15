#!/bin/bash
# This script runs the parameter_vector_creator.py with custom parameters

# Make sure the script is executable
chmod +x parameter_vector_creator.py

# Set project root directory and paths
PROJECT_ROOT="/homes/simondn/RashomonActiveLearning"
OUTPUT_DIR="$PROJECT_ROOT/Data/ParameterVectors"
mkdir -p "$OUTPUT_DIR"

# Default parameters
DATA="Iris"
SEED_START=0
SEED_END=49
TEST_PROPORTION=0.2
CANDIDATE_PROPORTION=0.8
SELECTOR_TYPE="BatchQBCDiversityFunction"
MODEL_TYPE="TreeFarmsFunction"
UNIQUE_ERRORS_INPUT="0 1"
N_ESTIMATORS=100
REGULARIZATION=0.01
RASHOMON_THRESHOLD_TYPE="Adder"
RASHOMON_THRESHOLD="0.025"
CLASSIFICATION_TYPE="Classification"
DIVERSITY_WEIGHT=0.4
BATCH_SIZE=10
PARTITION="short"
TIME_LIMIT="11:59:00"
MEMORY_LIMIT="30000M"
INCLUDE_RF=true
INCLUDE_RF_QBC=true

# Display help function
display_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --data VALUE                Dataset name (e.g., BankNote, Iris)"
  echo "  --seed-start VALUE          Start of seed range (default: 0)"
  echo "  --seed-end VALUE            End of seed range (default: 49)"
  echo "  --test-proportion VALUE     Proportion of data for testing (default: 0.2)"
  echo "  --candidate-proportion VALUE Proportion of data as candidates (default: 0.8)"
  echo "  --selector-type VALUE       Type of selector (default: BatchQBCDiversityFunction)"
  echo "  --model-type VALUE          Type of model (default: TreeFarmsFunction)"
  echo "  --unique-errors-input VALUES List of unique errors input values (default: '0 1')"
  echo "  --n-estimators VALUE        Number of estimators (default: 100)"
  echo "  --regularization VALUE      Regularization value (default: 0.01)"
  echo "  --rashomon-threshold-type VALUE Type of Rashomon threshold (default: Adder)"
  echo "  --rashomon-threshold VALUES List of Rashomon threshold values (default: '0.035 0.045')"
  echo "  --classification-type VALUE Type of classification (default: Classification)"
  echo "  --diversity-weight VALUE    Weight for diversity (default: 0.4)"
  echo "  --batch-size VALUE          Batch size for active learning (default: 10)"
  echo "  --partition VALUE           Partition type (default: short)"
  echo "  --time-limit VALUE          Time limit for runs (default: 11:59:00)"
  echo "  --memory-limit VALUE        Memory limit for runs (default: 30000M)"
  echo "  --no-random-forest          Exclude random forest passive learning simulations"
  echo "  --no-rf-qbc                 Exclude random forest QBC simulations"
  echo "  --output-dir VALUE          Directory to save parameter vector CSVs (default: ./parameter_vectors)"
  echo "  --help                      Display this help message"
  echo ""
  exit 0
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data)
      DATA="$2"
      shift 2
      ;;
    --seed-start)
      SEED_START="$2"
      shift 2
      ;;
    --seed-end)
      SEED_END="$2"
      shift 2
      ;;
    --test-proportion)
      TEST_PROPORTION="$2"
      shift 2
      ;;
    --candidate-proportion)
      CANDIDATE_PROPORTION="$2"
      shift 2
      ;;
    --selector-type)
      SELECTOR_TYPE="$2"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --unique-errors-input)
      UNIQUE_ERRORS_INPUT="$2"
      shift 2
      ;;
    --n-estimators)
      N_ESTIMATORS="$2"
      shift 2
      ;;
    --regularization)
      REGULARIZATION="$2"
      shift 2
      ;;
    --rashomon-threshold-type)
      RASHOMON_THRESHOLD_TYPE="$2"
      shift 2
      ;;
    --rashomon-threshold)
      RASHOMON_THRESHOLD="$2"
      shift 2
      ;;
    --classification-type)
      CLASSIFICATION_TYPE="$2"
      shift 2
      ;;
    --diversity-weight)
      DIVERSITY_WEIGHT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --time-limit)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --memory-limit)
      MEMORY_LIMIT="$2"
      shift 2
      ;;
    --no-random-forest)
      INCLUDE_RF=false
      shift
      ;;
    --no-rf-qbc)
      INCLUDE_RF_QBC=false
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help)
      display_help
      ;;
    *)
      echo "Unknown option: $1"
      display_help
      ;;
  esac
done

# Prepare RF options
if [ "$INCLUDE_RF" = false ]; then
  RF_OPTION="--no-random-forest"
else
  RF_OPTION=""
fi

if [ "$INCLUDE_RF_QBC" = false ]; then
  RF_QBC_OPTION="--no-rf-qbc"
else
  RF_QBC_OPTION=""
fi

echo "Creating parameter vector for dataset: $DATA"
echo "Using seed range: $SEED_START to $SEED_END"
echo "Output directory: $OUTPUT_DIR"

# Run the Python script with all parameters
./CreateParameterVector.py \
  --data "$DATA" \
  --seed-start "$SEED_START" \
  --seed-end "$SEED_END" \
  --test-proportion "$TEST_PROPORTION" \
  --candidate-proportion "$CANDIDATE_PROPORTION" \
  --selector-type "$SELECTOR_TYPE" \
  --model-type "$MODEL_TYPE" \
  --unique-errors-input $UNIQUE_ERRORS_INPUT \
  --n-estimators "$N_ESTIMATORS" \
  --regularization "$REGULARIZATION" \
  --rashomon-threshold-type "$RASHOMON_THRESHOLD_TYPE" \
  --rashomon-threshold $RASHOMON_THRESHOLD \
  --classification-type "$CLASSIFICATION_TYPE" \
  --diversity-weight "$DIVERSITY_WEIGHT" \
  --batch-size "$BATCH_SIZE" \
  --partition "$PARTITION" \
  --time-limit "$TIME_LIMIT" \
  --memory-limit "$MEMORY_LIMIT" \
  --output-dir "$OUTPUT_DIR" \
  $RF_OPTION $RF_QBC_OPTION

echo "Parameter vector created successfully!"