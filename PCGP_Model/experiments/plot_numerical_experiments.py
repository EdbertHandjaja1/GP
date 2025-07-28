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

    n_cols = len(test_functions)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6), sharey=False, sharex=False)

    if n_cols == 1:
        axes = [axes]

    for i, func in enumerate(test_functions):
        ax = axes[i]
        func_df = df[df['function'] == func]

        for model in models:
            model_df = func_df[func_df['model'] == model]
            scatter = ax.scatter(
                model_df['training_time'],
                model_df['rmse'],
                label=model,
                marker=markers[model],
                color=colors[model],
                s=model_df['n_train'] / 100 * 50 + 50,
                alpha=0.8
            )

        ax.set_title(f"Function: {func}")
        ax.set_xlabel("Training Time (seconds)")
        if i == 0:
            ax.set_ylabel("RMSE")

        ax.legend(title="Model")

    fig.suptitle("RMSE vs. Training Time for PCGP and Surmise Models", y=1.02, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plot_filename = os.path.join(plot_output_dir, 'rmse_vs_training_time_matplotlib.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")

    plt.show()

if __name__ == "__main__":
    results_file = 'experiments/output/numerical_results.csv'
    plot_output_directory = 'experiments/output/'
    plot_numerical_results(results_file, plot_output_directory)
