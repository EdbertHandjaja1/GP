import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel, generate_test_data

def plot_pcgp_testing_predictions(output_dim_to_plot=0):
    input_dim = 3
    output_dim = 5
    n_components = 3
    
    X_train, Y_train, X_test, Y_test, ranges, true_func = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim,
        function_type='multiplicative'
    )

    Y_train_single = Y_train[:, output_dim_to_plot].reshape(-1, 1)

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=1,
        input_dim=input_dim,
        output_dim=1
    )
    fitted_model = pcgp.fit(X_train, Y_train_single, ranges)

    Y_pred_mean, Y_pred_std = fitted_model.predict(X_test, ranges, return_std=True)
    
    plt.figure(figsize=(12, 7))
    
    sort_idx = np.argsort(X_test[:, 0])
    x_sorted = X_test[sort_idx, 0]
    y_true_sorted = Y_test[sort_idx, output_dim_to_plot]
    y_pred_sorted = Y_pred_mean[sort_idx, output_dim_to_plot]
    y_std_sorted = Y_pred_std[sort_idx, output_dim_to_plot]

    true_func_values = true_func(X_test[sort_idx])[:, output_dim_to_plot]

    print(np.round(
        np.c_[x_sorted,
                y_true_sorted,
                y_pred_sorted,
                y_pred_sorted - 2*y_std_sorted,
                y_pred_sorted + 2*y_std_sorted], 3)[:8])

    plt.scatter(x_sorted, y_true_sorted, c='black', marker='x', 
                s=100, label='True Test Values')
    
    plt.plot(x_sorted, y_pred_sorted, 'bo-', linewidth=2, 
             markersize=6, label='Predicted mean')
    plt.plot(x_sorted, true_func_values, 'r-', linewidth=3, 
             label='True function')
    noise_level = np.sqrt(pcgp.noise_var) * pcgp.standardization_scale
    plt.fill_between(x_sorted,
                    true_func_values - 2*noise_level,
                    true_func_values + 2*noise_level,
                    alpha=0.1, color='blue', label='True noise level')

    plt.xlabel('Input Dimension 1')
    plt.ylabel(f'Output Dimension {output_dim_to_plot + 1}')
    plt.title('PCGP AT Test Points')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    plot_pcgp_testing_predictions()
