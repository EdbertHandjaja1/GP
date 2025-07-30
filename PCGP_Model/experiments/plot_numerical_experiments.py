import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(results_filepath='experiments/output/numerical_results.csv', plot_output_dir='experiments/output/'):
    """
    Plots three analysis views of the experimental results:
    1) RMSE vs n_train
    2) Coverage vs n_train
    3) RMSE vs Training Time
    
    Args:
        results_filepath (str): Path to the CSV file containing the numerical results.
        plot_output_dir (str): Directory where the plots will be saved.
    """
    df = pd.read_csv(results_filepath)
    
    os.makedirs(plot_output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    markers = {'PCGP': 'o', 'Surmise': 's'}
    colors = {'PCGP': 'tab:blue', 'Surmise': 'tab:orange'}
    line_styles = {'PCGP': '-', 'Surmise': '--'}
    
    test_functions = df['function'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # plot 1: RMSE vs n_train
    ax1 = axes[0]
    for func in test_functions:
        for model in ['PCGP', 'Surmise']:
            subset = df[(df['function'] == func) & (df['model'] == model)]
            ax1.plot(subset['n_train'], subset['rmse'], 
                    marker=markers[model], color=colors[model], 
                    linestyle=line_styles[model], 
                    label=f"{model} - {func}")
    
    ax1.set_xlabel('Training Sample Size (n)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Training Sample Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, which="both", ls="-")
    
    # plot 2: Coverage vs n_train 
    ax2 = axes[1]
    for func in test_functions:
        for model in ['PCGP', 'Surmise']:
            subset = df[(df['function'] == func) & (df['model'] == model)]
            ax2.plot(subset['n_train'], subset['ci_coverage'], 
                    marker=markers[model], color=colors[model], 
                    linestyle=line_styles[model], 
                    label=f"{model} - {func}")
    
    ax2.axhline(95, color='red', linestyle=':', label='Target 95% Coverage')
    ax2.set_xlabel('Training Sample Size (n)')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('CI Coverage vs Training Sample Size')
    ax2.set_xscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, which="both", ls="-")
    
    # plot 3: RMSE vs Training Time
    ax3 = axes[2]
    for func in test_functions:
        for model in ['PCGP', 'Surmise']:
            subset = df[(df['function'] == func) & (df['model'] == model)]
            ax3.scatter(subset['training_time'], subset['rmse'], 
                       marker=markers[model], color=colors[model], 
                       s=subset['n_train']/50, 
                       alpha=0.7, 
                       label=f"{model} - {func}")
    
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE vs Training Time (size = n_train)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, which="both", ls="-")
    
    plt.tight_layout()
    
    plot_filename = os.path.join(plot_output_dir, 'combined_analysis.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_filename}")
    
    plt.show()

def plot_rmse_boxplots(results_filepath='experiments/output/numerical_results.csv', 
                      plot_output_dir='experiments/output/'):
    """
    Creates a single box plot showing RMSE distributions for PCGP vs Surmise across different sample sizes.
    """
    df = pd.read_csv(results_filepath)
    
    os.makedirs(plot_output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    sample_sizes = sorted(df['n_train'].unique())
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    box_data = []
    box_labels = []
    box_colors = []
    
    colors = {'PCGP': 'lightblue', 'Surmise': 'lightgreen'}
    
    for n in sample_sizes:
        subset = df[df['n_train'] == n]
        
        pcgp_data = subset[subset['model'] == 'PCGP']['rmse']
        surmise_data = subset[subset['model'] == 'Surmise']['rmse']
        
        box_data.extend([pcgp_data, surmise_data])
        box_labels.extend([f'PCGP\nn={n}', f'Surmise\nn={n}'])
        box_colors.extend([colors['PCGP'], colors['Surmise']])
    
    boxprops = dict(linestyle='-', linewidth=1.5)
    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
    bp = ax.boxplot(box_data,
                   patch_artist=True,
                   labels=box_labels,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   widths=0.6)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    
    for i in range(1, len(sample_sizes)):
        ax.axvline(x=i*2 + 0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_xlabel('Model and Sample Size', fontsize=12)
    ax.set_title('RMSE Distribution Comparison: PCGP vs Surmise Across Sample Sizes', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    plot_filename = os.path.join(plot_output_dir, 'rmse_boxplots_combined.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Combined box plots saved to {plot_filename}")
    
    plt.show()

if __name__ == "__main__":
    results_file = 'experiments/output/numerical_results_20250729_154234.csv'
    plot_output_directory = 'experiments/output/'
    # plot_results(results_file, plot_output_directory)
    plot_rmse_boxplots(results_file, plot_output_directory)