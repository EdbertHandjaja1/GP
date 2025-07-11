import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import generate_test_data

from surmise.emulation import emulator
from surmise.calibration import calibrator

def plot_surmise_testing_predictions(output_dim_to_plot=0):
    """Plot Surmise emulator predictions vs true values"""
    input_dim = 3
    output_dim = 5
    
    X_train, Y_train, X_test, Y_test, ranges, true_func = generate_test_data(
        input_dim=input_dim,
        output_dim=output_dim,
        )

    theta_train = Y_train.T               

    emu = emulator(
        x=X_train, 
        theta=theta_train, 
        f=Y_train, 
        method='PCGP')

    emu.fit()

    pred = emu.predict(x=X_test, theta=theta_train)   

    Y_pred_mean = pred.mean()           
    Y_pred_std  = np.sqrt(pred.var())
        
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
                s=100, label='True Test Values', zorder=5)
    
    plt.plot(x_sorted, y_pred_sorted, 'bo-', linewidth=2,
             markersize=6, label='Surmise Predicted Mean', zorder=4)
    
    plt.plot(x_sorted, true_func_values, 'r-', linewidth=3,
             label='True Function', zorder=3)
    
    plt.fill_between(x_sorted,
                     (y_pred_sorted - 2 * y_std_sorted),
                     (y_pred_sorted + 2 * y_std_sorted),
                     alpha=0.2, color='blue', 
                     label='95% Confidence Interval', zorder=2)
    
    plt.xlabel('Input Dimension 1')
    plt.ylabel(f'Output Dimension {output_dim_to_plot + 1}')
    plt.title('Surmise Emulator Predictions at Test Points')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_surmise_testing_predictions(output_dim_to_plot=0)