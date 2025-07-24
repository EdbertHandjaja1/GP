import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib

def plot_numerical_results(results_filepath='experiments/output/numerical_results.csv', plot_output_dir='experiments/output/'):
    """
    Plots the RMSE vs. Training Time for PCGP and Surmise models based on numerical experiment results,
    using only Matplotlib.

    Args:
        results_filepath (str): Path to the CSV file containing the numerical results.
        plot_output_dir (str): Directory where the plots will be saved.
    """

    df = pd.read_csv(results_filepath)

    plt.style.use('seaborn-v0_8-whitegrid') 

    test_functions = df['function'].unique()
    models = df['model'].unique()

    markers = {'PCGP': 'o', 'Surmise': 's'}
    colors = {'PCGP': 'tab:blue', 'Surmise': 'tab:orange'}

    pass

if __name__ == "__main__":
    results_file = 'experiments/output/numerical_results.csv'
    plot_output_directory = 'experiments/output/'
    plot_numerical_results(results_file, plot_output_directory)