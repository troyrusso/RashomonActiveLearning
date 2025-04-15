#!/usr/bin/env python3
"""
Parameter Vector Creator

This script automates the creation of parameter vectors for machine learning simulations.
It generates combinations of parameters and outputs them to a CSV file.
"""

import argparse
import itertools
import os
import pandas as pd


def create_parameter_vector(data, 
                            seed_range=None, 
                            test_proportion=0.2, 
                            candidate_proportion=0.8,
                            selector_type="BatchQBCDiversityFunction", 
                            model_type="TreeFarmsFunction",
                            unique_errors_input=None, 
                            n_estimators=100, 
                            regularization=0.01,
                            rashomon_threshold_type="Adder", 
                            rashomon_threshold=None,
                            classification_type="Classification", 
                            diversity_weight=0.4, 
                            batch_size=10,
                            partition="short", time_limit="11:59:00", 
                            memory_limit="30000M",
                            include_random_forest=True, 
                            include_rf_qbc=True, 
                            output_dir=None):
    """
    Create a parameter vector for machine learning simulations.
    
    Parameters
    ----------
    data : str
        The name of the dataset to use
    seed_range : tuple, optional
        Range of seeds to use (start, end)
    test_proportion : float, optional
        Proportion of data for testing
    candidate_proportion : float, optional
        Proportion of data used as candidates
    selector_type : str, optional
        Type of selector to use
    model_type : str, optional
        Type of model to use
    unique_errors_input : list, optional
        List of unique errors input values
    n_estimators : int, optional
        Number of estimators
    regularization : float, optional
        Regularization value
    rashomon_threshold_type : str, optional
        Type of Rashomon threshold
    rashomon_threshold : list, optional
        List of Rashomon threshold values
    classification_type : str, optional
        Type of classification
    diversity_weight : float, optional
        Weight for diversity in active learning
    batch_size : int, optional
        Batch size for active learning
    partition : str, optional
        Partition type
    time_limit : str, optional
        Time limit for runs
    memory_limit : str or int, optional
        Memory limit for runs
    include_random_forest : bool, optional
        Whether to include random forest passive learning simulations
    include_rf_qbc : bool, optional
        Whether to include random forest QBC simulations
    output_dir : str, optional
        Directory to save the parameter vector CSV file
    
    Returns
    -------
    pandas.DataFrame
        The generated parameter vector
    """
    # Abbreviation dictionary for dataset names
    abbreviation_dictionary = {
        "BankNote": "BN",
        "Bar7": "B7",
        "BreastCancer": "BC",
        "CarEvaluation": "CE",
        "COMPAS": "CP",
        "FICO": "FI",
        "Haberman": "HM",
        "Iris": "IS",
        "MONK1": "M1",
        "MONK3": "M3"
    }
    
    job_name_abbrev = abbreviation_dictionary[data]
    
    # Set default values if not provided
    if seed_range is None:
        return "No seed given"
    else:
        seeds = list(range(seed_range[0], seed_range[1] + 1))
    
    if unique_errors_input is None:
        unique_errors_input = [0, 1]
    elif not isinstance(unique_errors_input, list):
        unique_errors_input = [unique_errors_input]
    
    if rashomon_threshold is None:
        return "No rashomon threshold given"
    elif not isinstance(rashomon_threshold, list):
        rashomon_threshold = [rashomon_threshold]
        
    # Main parameter dictionary
    parameter_dictionary = {
        "Data": [data],
        "Seed": seeds,
        "TestProportion": [test_proportion],
        "CandidateProportion": [candidate_proportion],
        "SelectorType": [selector_type],
        "ModelType": [model_type],
        "UniqueErrorsInput": unique_errors_input,
        "n_estimators": [n_estimators],
        "regularization": [regularization],
        "RashomonThresholdType": [rashomon_threshold_type],
        "RashomonThreshold": rashomon_threshold,
        "Type": [classification_type],
        "DiversityWeight": [diversity_weight],
        "BatchSize": [batch_size],
        "Partition": [partition],
        "Time": [time_limit],
        "Memory": [memory_limit]
    }
    
    # Create parameter vector
    parameter_vector = pd.DataFrame.from_records(
        itertools.product(*parameter_dictionary.values()),
        columns=parameter_dictionary.keys()
    )
    
    # Include passive learning (Random Forest) if requested
    if include_random_forest:
        rf_parameter_dictionary = {
            "Data": [data],
            "Seed": list(range(0, 50)),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearning"],
            "ModelType": ["RandomForestClassificationFunction"],
            "UniqueErrorsInput": [1],
            "n_estimators": [100],
            "regularization": [0.01],
            "RashomonThresholdType": ["Adder"],
            "RashomonThreshold": [0],
            "Type": ["Classification"],
            "DiversityWeight": [0],
            "BatchSize": [batch_size],
            "Partition": ["short"],
            "Time": ["00:59:00"],
            "Memory": [1000]
        }
        
        rf_parameter_vector = pd.DataFrame.from_records(
            itertools.product(*rf_parameter_dictionary.values()),
            columns=rf_parameter_dictionary.keys()
        )
        
        parameter_vector = pd.concat([parameter_vector, rf_parameter_vector])
    
    # Include Random Forest QBC simulations if requested
    if include_rf_qbc:
        rf_qbc_parameter_dictionary = {
            "Data": [data],
            "Seed": list(range(0, 50)),
            "TestProportion": [0.2],
            "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCDiversityFunction"],
            "ModelType": ["RandomForestClassificationFunction"],
            "UniqueErrorsInput": [0],
            "n_estimators": [100],
            "regularization": [0.01],
            "RashomonThresholdType": ["Adder"],
            "RashomonThreshold": [0],
            "Type": ["Classification"],
            "DiversityWeight": [diversity_weight],
            "BatchSize": [batch_size],
            "Partition": ["short"],
            "Time": ["00:59:00"],
            "Memory": [1000]
        }
        
        rf_qbc_parameter_vector = pd.DataFrame.from_records(
            itertools.product(*rf_qbc_parameter_dictionary.values()),
            columns=rf_qbc_parameter_dictionary.keys()
        )
        
        parameter_vector = pd.concat([parameter_vector, rf_qbc_parameter_vector])
    
    # Sort by seed and reset index
    parameter_vector = parameter_vector.sort_values("Seed")
    parameter_vector.index = range(0, parameter_vector.shape[0])
    
    # Generate job names
    parameter_vector["JobName"] = (
        parameter_vector["Seed"].astype(str) +
        job_name_abbrev +
        "_MT" + parameter_vector["ModelType"].astype(str) +
        "_UEI" + parameter_vector["UniqueErrorsInput"].astype(str) +
        "_" + parameter_vector["RashomonThresholdType"].astype(str) +
        parameter_vector["RashomonThreshold"].astype(str) +
        "_D" + parameter_vector["DiversityWeight"].astype(str) +
        "B" + parameter_vector["BatchSize"].astype(str)
    )
    
    # Replace job names with more concise versions
    parameter_vector["JobName"] = (
        parameter_vector["JobName"]
        .str.replace(r"_MTTreeFarmsFunction_UEI0_", "_D", regex=True)
        .str.replace(r"_MTTreeFarmsFunction_UEI1_", "_U", regex=True)
        .str.replace(r"Adder", "A", regex=True)
        .str.replace(r"Multiplier", "M", regex=True)
        .str.replace(r"_MTRandomForestClassificationFunction_UEI0_", "_RF", regex=True)
        .str.replace(r"_MTRandomForestClassificationFunction_UEI1_", "_PL", regex=True)
        .str.replace(r"0.", "", regex=False)
    )
    
    # Generate output paths
    parameter_vector["Output"] = (
        parameter_vector["Data"].astype(str) + "/" +
        parameter_vector["ModelType"].astype(str) + "/Raw/" +
        parameter_vector["JobName"] + ".pkl"
    )
    parameter_vector["Output"] = parameter_vector["Output"].str.replace("Function", "", regex=False)
    
    # Save parameter vector to CSV if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ParameterVector{data}.csv")
        parameter_vector.to_csv(output_path, index=False)
        print(f"Parameter vector saved to {output_path}")
    
    return parameter_vector


def main():
    """Main function to parse arguments and create parameter vector."""
    parser = argparse.ArgumentParser(description="Create parameter vectors for ML simulations")
    parser.add_argument("--data", required=True, help="Dataset name (e.g., BankNote, Iris)")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed value")
    parser.add_argument("--seed-end", type=int, default=49, help="Ending seed value (inclusive)")
    parser.add_argument("--test-proportion", type=float, default=0.2, help="Proportion of data for testing")
    parser.add_argument("--candidate-proportion", type=float, default=0.8, help="Proportion of data used as candidates")
    parser.add_argument("--selector-type", default="BatchQBCDiversityFunction", help="Type of selector to use")
    parser.add_argument("--model-type", default="TreeFarmsFunction", help="Type of model to use")
    parser.add_argument("--unique-errors-input", type=int, nargs="+", default=[0, 1], help="List of unique errors input values")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--regularization", type=float, default=0.01, help="Regularization value")
    parser.add_argument("--rashomon-threshold-type", default="Adder", help="Type of Rashomon threshold")
    parser.add_argument("--rashomon-threshold", type=float, nargs="+", default=[0.035, 0.045], help="List of Rashomon threshold values")
    parser.add_argument("--classification-type", default="Classification", help="Type of classification")
    parser.add_argument("--diversity-weight", type=float, default=0.4, help="Weight for diversity in active learning")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for active learning")
    parser.add_argument("--partition", default="short", help="Partition type")
    parser.add_argument("--time-limit", default="11:59:00", help="Time limit for runs")
    parser.add_argument("--memory-limit", default="30000M", help="Memory limit for runs")
    parser.add_argument("--no-random-forest", action="store_true", help="Exclude random forest passive learning simulations")
    parser.add_argument("--no-rf-qbc", action="store_true", help="Exclude random forest QBC simulations")
    parser.add_argument("--output-dir", default="./parameter_vectors", help="Directory to save parameter vector CSV")
    
    args = parser.parse_args()
    
    create_parameter_vector(
        data=args.data,
        seed_range=(args.seed_start, args.seed_end),
        test_proportion=args.test_proportion,
        candidate_proportion=args.candidate_proportion,
        selector_type=args.selector_type,
        model_type=args.model_type,
        unique_errors_input=args.unique_errors_input,
        n_estimators=args.n_estimators,
        regularization=args.regularization,
        rashomon_threshold_type=args.rashomon_threshold_type,
        rashomon_threshold=args.rashomon_threshold,
        classification_type=args.classification_type,
        diversity_weight=args.diversity_weight,
        batch_size=args.batch_size,
        partition=args.partition,
        time_limit=args.time_limit,
        memory_limit=args.memory_limit,
        include_random_forest=not args.no_random_forest,
        include_rf_qbc=not args.no_rf_qbc,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()