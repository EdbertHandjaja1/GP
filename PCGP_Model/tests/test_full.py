import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def plot_pcgp_results(output_dim_to_plot=0):
    input_dim = 3
    output_dim = 5
    n_components = 3
    
    X_train, Y_train, _, _, ranges, true_func = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim,
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=output_dim
    )
    fitted_model = pcgp.fit(X_train, Y_train, ranges)

    for input_dim_to_vary in range(input_dim):
        X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        
        X_plot_full = np.tile(np.median(X_train, axis=0), (100, 1))
        X_plot_full[:, input_dim_to_vary] = X_plot[:, 0]

        Y_pred_mean, Y_pred_std = fitted_model.predict(X_plot_full, ranges, return_std=True)
        Y_true = true_func(X_plot_full)
        
        mean_plot = Y_pred_mean[:, output_dim_to_plot]
        std_plot = Y_pred_std[:, output_dim_to_plot]

        plt.figure(figsize=(12, 7))
        
        plt.scatter(X_train[:, input_dim_to_vary], Y_train[:, output_dim_to_plot], 
                    c='black', marker='x', label='Training Data')
        
        plt.plot(X_plot, Y_true[:, output_dim_to_plot], 'r-', linewidth=2, 
                 label='True Function')

        plt.plot(X_plot, mean_plot, 'b--', linewidth=2, 
                 label='Predicted Mean')

        plt.fill_between(X_plot.flatten(),
                 (mean_plot - 2 * std_plot),
                 (mean_plot + 2 * std_plot),
                 alpha=0.2, color='blue', label='95% Confidence Interval')

        plt.xlabel(f'Input Dimension {input_dim_to_vary + 1}')
        plt.ylabel(f'Output Dimension {output_dim_to_plot + 1}')
        plt.title(f'PCGP: Input Dimension {input_dim_to_vary + 1}, Output Dimension {output_dim_to_plot + 1}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


plot_pcgp_results()