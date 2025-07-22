import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from testfunc_wrapper import TestFuncCaller

def run_pcgp_experiment(function_name='borehole', output_dim=0, n_samples=100, noise_level=0.05):
    func_caller = TestFuncCaller(function_name)
    meta = func_caller.info
    
    X_train = np.random.uniform(0, 1, (n_samples, meta['xdim']))
    
    if meta['thetadim'] > 0:
        theta_train = np.random.uniform(0, 1, (n_samples, meta['thetadim']))
    else:
        theta_train = None
    
    Y_train = func_caller.info['nofailmodel'](X_train, theta_train)
    Y_true = func_caller.info['true_func'](X_train)
    
    Y_train += noise_level * np.std(Y_true) * np.random.randn(*Y_train.shape)
    
    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=1,
        input_dim=meta['xdim'],
        output_dim=1
    )
    
    ranges = np.column_stack([np.zeros(meta['xdim']), np.ones(meta['xdim'])])
    
    fitted_model = pcgp.fit(
        X_train, 
        Y_train[:, output_dim].reshape(-1, 1), 
        ranges
    )
    
    Y_pred_mean, Y_pred_std = fitted_model.predict(X_train, ranges, return_std=True)
    
    train_rmse = np.sqrt(np.mean((Y_pred_mean - Y_train[:, output_dim].reshape(-1, 1)) ** 2))
    
    plt.figure(figsize=(12, 7))
    sort_idx = np.argsort(X_train[:, 0])
    
    plt.scatter(X_train[sort_idx, 0], Y_train[sort_idx, output_dim], 
                c='black', marker='x', s=100, label='Noisy Training Values')
    plt.plot(X_train[sort_idx, 0], Y_pred_mean[sort_idx, 0], 'bo-', 
             linewidth=2, markersize=6, label='Predicted mean')
    plt.plot(X_train[sort_idx, 0], Y_true[sort_idx, output_dim], 'r-', 
             linewidth=3, label='True function')
    plt.fill_between(X_train[sort_idx, 0],
                    (Y_pred_mean[sort_idx, 0] - 2 * Y_pred_std[sort_idx, 0]),
                    (Y_pred_mean[sort_idx, 0] + 2 * Y_pred_std[sort_idx, 0]),
                    alpha=0.2, color='blue', label='95% Confidence Interval')
    
    plt.xlabel('First Input Dimension')
    plt.ylabel(f'Output (Dimension {output_dim + 1})')
    plt.title(f'PCGP Results for {meta["function"]} Function')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    print("\n" + "="*50)
    print(f"{meta['function']} Function Results".center(50))
    print("="*50)
    print(f"{'Training RMSE:':<20}{train_rmse:.4f}")
    print("="*50 + "\n")
    
    plt.show()

if __name__ == "__main__":
    run_pcgp_experiment(function_name='borehole')
    # run_pcgp_experiment(function_name='otlcircuit')
    # run_pcgp_experiment(function_name='piston')
    # run_pcgp_experiment(function_name='wingweight')