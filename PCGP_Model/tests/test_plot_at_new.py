import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def plot_pcgp_testing_predictions(output_dim_to_plot=0):
    input_dim = 3
    output_dim = 5
    n_components = 3
    
    X_train, Y_train, X_test, Y_test, ranges, true_func = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim,
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=output_dim
    )
    fitted_model = pcgp.fit(X_train, Y_train, ranges)

    # testing points
    Y_pred_mean, Y_pred_std = fitted_model.predict(X_test, ranges, return_std=True)
    
    plt.figure(figsize=(12, 7))
    
    sort_idx = np.argsort(X_test[:, 0])
    x_sorted = X_test[sort_idx, 0]
    y_true_sorted = Y_test[sort_idx, output_dim_to_plot]
    y_pred_sorted = Y_pred_mean[sort_idx, output_dim_to_plot]
    y_std_sorted = Y_pred_std[sort_idx, output_dim_to_plot]

    plt.scatter(x_sorted, y_true_sorted, c='black', marker='x', 
                s=100, label='True Test Values')
    
    plt.plot(x_sorted, y_pred_sorted, 'bo-', linewidth=2, 
             markersize=6, label='Predicted mean')
    plt.fill_between(x_sorted,
             (y_pred_sorted - 2 * y_std_sorted),
             (y_pred_sorted + 2 * y_std_sorted),
             alpha=0.2, color='blue', label='95% Confidence Interval')

    plt.xlabel('Input Dimension 1')
    plt.ylabel(f'Output Dimension {output_dim_to_plot + 1}')
    plt.title('PCGP AT Test Points')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

plot_pcgp_testing_predictions()