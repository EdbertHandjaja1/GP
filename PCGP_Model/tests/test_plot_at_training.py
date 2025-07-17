import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel, generate_test_data

def plot_pcgp_training_predictions(output_dim_to_plot=0):
    """
    Test function to plot PCGP predictions at training points.
    
    Args:
        output_dim_to_plot (int): Which output dimension to visualize (0-based index)
    """
    input_dim = 3
    output_dim = 5
    
    X_train, Y_train, _, Y_test, ranges, true_func = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim,
        function_type='default'
    )

    Y_train_single = Y_train[:, output_dim_to_plot].reshape(-1, 1)

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=1,
        input_dim=input_dim,
        output_dim=1
    )
    fitted_model = pcgp.fit(X_train, Y_train_single, ranges)

    Y_pred_mean, Y_pred_std = fitted_model.predict(X_train, ranges, return_std=True)
    
    train_rmse = np.sqrt(np.mean((Y_pred_mean - Y_train_single) ** 2))

    print("\n" + "="*50)
    print(f"{'RMSE':^50}")
    print("="*50)
    print(f"{'RMSE:':<20}{train_rmse:.4f}")
    print("="*50 + "\n")

    plt.figure(figsize=(12, 7))
    
    sort_idx = np.argsort(X_train[:, 0])
    x_sorted = X_train[sort_idx, 0]
    y_true_sorted = Y_train[sort_idx, output_dim_to_plot]
    y_pred_sorted = Y_pred_mean[sort_idx, output_dim_to_plot]
    y_std_sorted = Y_pred_std[sort_idx, output_dim_to_plot]
    true_func_values = true_func(X_train[sort_idx])[:, output_dim_to_plot]

    print(np.round(
        np.c_[x_sorted,
                y_true_sorted,
                y_pred_sorted,
                y_pred_sorted - 2*y_std_sorted,
                y_pred_sorted + 2*y_std_sorted], 3)[:8])

    plt.scatter(x_sorted, y_true_sorted, c='black', marker='x', 
                s=100, label='Noisy Training Values')
    
    plt.plot(x_sorted, y_pred_sorted, 'bo-', linewidth=2, 
             markersize=6, label='Predicted mean')
    plt.plot(x_sorted, true_func_values, 'r-', linewidth=3, 
             label='True function')
    # plt.fill_between(x_sorted,
    #          (y_pred_sorted - 2 * y_std_sorted),
    #          (y_pred_sorted + 2 * y_std_sorted),
    #          alpha=0.2, color='blue', label='95% Confidence Interval')

    noise_level = np.sqrt(pcgp.noise_var) * pcgp.standardization_scale
    plt.fill_between(x_sorted,
                    true_func_values - 2*noise_level,
                    true_func_values + 2*noise_level,
                    alpha=0.1, color='blue', label='True noise level')

    plt.xlabel('Input Dimension 1')
    plt.ylabel(f'Output Dimension {output_dim_to_plot + 1}')
    plt.title('PCGP Predictions at Training Points')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_pcgp_training_predictions()