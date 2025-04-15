#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Analysis Script for Rashomon Set Active Learning
Takes DataType and Value as command-line arguments
"""

import os
import argparse
import sys
import itertools
import numpy as np
import math as math
import pandas as pd 
import random as random

# Try to import matplotlib - if it fails, provide installation instructions
try:
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
except ImportError as e:
    module_name = str(e).split("'")[-2]
    print(f"Error: Missing required module '{module_name}'")
    print("\nPlease install the missing module with:")
    print(f"  pip install {module_name}")
    print("\nIf you're using a virtual environment, make sure it's activated:")
    print("  source /path/to/your/venv/bin/activate")
    sys.exit(1)

# Add the project root to the path so we can import from utils
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
try:
    from utils.Auxiliary import *
except ImportError:
    print("Error: Cannot import from utils.Auxiliary")
    print("Make sure you're running this script from the Code/Analysis directory")
    sys.exit(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze Rashomon Set Active Learning data')
    parser.add_argument('DataType', type=str, help='Type of data to analyze (e.g., "BankNote")')
    parser.add_argument('Value', type=str, help='Value parameter for TreeFarms (e.g., "025")')
    parser.add_argument('--BaseDirectory', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            '../../Results/'),
                        help='Base directory for results')
    parser.add_argument('--output_dir', type=str, 
                        default=None,  # Will be set based on DataType below
                        help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Set default output directory based on DataType if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.BaseDirectory, args.DataType, 
                                    'TreeFarms', 'ProcessedResults')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    print(f"Analyzing {args.DataType} dataset with TreeFarms value {args.Value}")
    
    try:
        PassiveLearningRF = LoadAnalyzedData(args.DataType, args.BaseDirectory, "RandomForestClassification", "PLA0")
        RandomForestResults = LoadAnalyzedData(args.DataType, args.BaseDirectory, "RandomForestClassification", "RFA0")
        AnalyzedDataUNREALDUREAL = LoadAnalyzedData(args.DataType, args.BaseDirectory, "TreeFarms", args.Value)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please check that the data files exist for {args.DataType} with value {args.Value}")
        sys.exit(1)
    
    # Report shape information
    ShapeTable = {"PassiveLearningRF": PassiveLearningRF["Error"].shape[0],
                  "RandomForestResults": RandomForestResults["Error"].shape[0],
                  "DUREAL":[AnalyzedDataUNREALDUREAL["Error_DUREAL"].shape[0]],
                  "UNREAL": [AnalyzedDataUNREALDUREAL["Error_UNREAL"].shape[0]]}
    ShapeTable = pd.DataFrame(ShapeTable)
    print("\nShape Information:")
    print(ShapeTable)
    
    # Report runtime information
    TimeTable = {
        "DUREAL Mean":[str(round(np.mean(AnalyzedDataUNREALDUREAL["Time_DUREAL"])/60,2))],
        "UNREAL Mean": [str(round(np.mean(AnalyzedDataUNREALDUREAL["Time_UNREAL"])/60,2))],
        "DUREAL max":[str(round(np.max(AnalyzedDataUNREALDUREAL["Time_DUREAL"])/60,2))],
        "UNREAL max": [str(round(np.max(AnalyzedDataUNREALDUREAL["Time_UNREAL"])/60,2))]
    }
    TimeTable = pd.DataFrame(TimeTable)
    print("\nRuntime Information (minutes):")
    print(TimeTable)
    
    # Set up plotting parameters
    PlotSubtitle = f"Dataset: {args.DataType} (Value: {args.Value})"
    colors = {
        "PassiveLearning": "black",
        "RandomForest": "green",
        "DUREAL": "orange",
        "UNREAL": "blue"
    }
    
    linestyles = {
        "PassiveLearning": "solid",
        "RandomForest": "solid",
        "DUREAL": "solid",
        "UNREAL": "solid"
    }
    
    LegendMapping = {
        "DUREAL0": f"DUREAL (ε = 0.{args.Value})",
        "UNREAL0": f"UNREAL (ε = 0.{args.Value})",
    }
    
    # Create subdirectories for organization
    error_dir = os.path.join(args.output_dir, "ErrorVec")
    tree_counts_dir = os.path.join(args.output_dir, "TreeCount")
    stats_dir = os.path.join(args.output_dir, "Analysis")
    
    for directory in [error_dir, tree_counts_dir, stats_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Generate Error Plot
    try:
        error_plot = MeanVariancePlot(
            RelativeError = None,
            PassiveLearning = PassiveLearningRF["Error"],
            RandomForest = RandomForestResults["Error"],
            DUREAL = AnalyzedDataUNREALDUREAL["Error_DUREAL"],
            UNREAL = AnalyzedDataUNREALDUREAL["Error_UNREAL"],
            Colors = colors,
            LegendMapping = LegendMapping,
            Linestyles = linestyles,
            Y_Label = "F1 Score",
            Subtitle = PlotSubtitle,
            TransparencyVal = 0.05,
            VarInput = True,
            CriticalValue = 1.96
        )
        
        # Save the error plot
        error_plot_path = os.path.join(error_dir, f"{args.DataType}_error_plot_{args.Value}.png")
        error_plot.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved error plot to: {error_plot_path}")
    except Exception as e:
        print(f"Error generating error plot: {e}")
    
    # Generate Tree Counts Plot
    try:
        tree_counts_plot = MeanVariancePlot(
            RelativeError = None,
            DUREAL = np.log(AnalyzedDataUNREALDUREAL["TreeCounts_ALL_UNREAL"]),
            UNREAL = np.log(AnalyzedDataUNREALDUREAL["TreeCounts_UNIQUE_UNREAL"]),
            Colors = colors,
            LegendMapping = LegendMapping,
            Linestyles = linestyles,
            Y_Label = "log(Number of Trees in the Rashomon Set)",
            Subtitle = PlotSubtitle,
            TransparencyVal = 0.05,
            VarInput = False,
            CriticalValue = 1.96
        )
        
        # Save the tree counts plot
        tree_counts_plot_path = os.path.join(tree_counts_dir, f"{args.DataType}_tree_counts_{args.Value}.png")
        tree_counts_plot.savefig(tree_counts_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved tree counts plot to: {tree_counts_plot_path}")
    except Exception as e:
        print(f"Error generating tree counts plot: {e}")
    
    # Run Wilcoxon Ranked Signed Test
    try:
        WRSTResults = WilcoxonRankSignedTest({
            "PassiveLearning": PassiveLearningRF["Error"],
            "RandomForest": RandomForestResults["Error"],
            "UNREAL": AnalyzedDataUNREALDUREAL["Error_UNREAL"],
            "DUREAL": AnalyzedDataUNREALDUREAL["Error_DUREAL"]},
            5
        )
        
        # Save Wilcoxon test results to CSV and LaTeX
        wrst_csv_path = os.path.join(stats_dir, f"{args.DataType}_wilcoxon_results_{args.Value}.csv")
        WRSTResults.to_csv(wrst_csv_path)
        
        wrst_latex_path = os.path.join(stats_dir, f"{args.DataType}_wilcoxon_results_{args.Value}.tex")
        with open(wrst_latex_path, 'w') as f:
            f.write(WRSTResults.to_latex())
        
        print(f"Saved Wilcoxon test results to CSV: {wrst_csv_path}")
        print(f"Saved Wilcoxon test results to LaTeX: {wrst_latex_path}")
        print("\nWilcoxon Ranked Signed Test Results:")
        print(WRSTResults)
    except Exception as e:
        print(f"Error generating Wilcoxon test: {e}")

if __name__ == "__main__":
    main()