import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator

def run_experiments(n_samples=100, noise_level=0.05, output_dim=0):
    test_functions = ['borehole', 'otlcircuit', 'piston', 'wingweight']
    models = ['pcgp', 'surmise']
    
    for function_name in test_functions:
        func_caller = TestFuncCaller(function_name)
        meta = func_caller.info
        
        # generate training data once for both models
        theta_train = np.random.uniform(0, 1, (n_samples, meta['thetadim'])) 
        X_train = np.random.uniform(0, 1, (n_samples, meta['xdim']))
        
        Y_train = func_caller.info['nofailmodel'](X_train, theta_train)
        Y_true = func_caller.info['true_func'](X_train)
        
        Y_train += noise_level * np.std(Y_train) 
        
        plt.figure(figsize=(15, 8))
        
        sort_idx = np.argsort(X_train[:, 0])
        
        plt.scatter(X_train[sort_idx, 0], Y_train[sort_idx, output_dim], 
                   c='black', marker='x', s=100, label='Noisy Training Values', alpha=0.7)
        plt.plot(X_train[sort_idx, 0], Y_true[sort_idx, output_dim], 'r-', 
                linewidth=3, label='True function')
        
        results = {}
        
        for model in models:
            if model == 'pcgp':
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
                
                plt.plot(X_train[sort_idx, 0], Y_pred_mean[sort_idx, 0], 'b-', 
                        linewidth=3, label='PCGP Predicted mean')
                plt.fill_between(X_train[sort_idx, 0],
                                (Y_pred_mean[sort_idx, 0] - 2 * Y_pred_std[sort_idx, 0]),
                                (Y_pred_mean[sort_idx, 0] + 2 * Y_pred_std[sort_idx, 0]),
                                alpha=0.2, color='blue', label='PCGP 95% CI')
                
                results['pcgp'] = train_rmse
                
                print("\n" + "="*50)
                print(f"PCGP: {meta['function']} Function Results".center(50))
                print("="*50)
                print(f"{'Training RMSE:':<20}{train_rmse:.4f}")
                print("="*50)
            
            else:  
                theta_emu_train = X_train 
                x_emu_train = np.array([[0]])
                f_emu_train = Y_train[:, output_dim].reshape(1, -1) 
                
                emu = emulator(
                    x=x_emu_train, 
                    theta=theta_emu_train, 
                    f=f_emu_train, 
                    method='PCGP',
                    options={'epsilon': 0})
                emu.fit()
                
                pred = emu.predict(x=x_emu_train, theta=X_train)
                Y_pred_mean = pred.mean().flatten()
                Y_pred_std = np.sqrt(pred.var()).flatten()
                train_rmse = np.sqrt(np.mean((Y_pred_mean - Y_train[:, output_dim]) ** 2))
                
                plt.plot(X_train[sort_idx, 0], Y_pred_mean[sort_idx], 'g--', 
                        linewidth=3, label='Surmise Predicted mean')
                plt.fill_between(X_train[sort_idx, 0],
                                (Y_pred_mean[sort_idx] - 2 * Y_pred_std[sort_idx]),
                                (Y_pred_mean[sort_idx] + 2 * Y_pred_std[sort_idx]),
                                alpha=0.15, color='green', label='Surmise 95% CI')
                
                results['surmise'] = train_rmse
                
                print("\n" + "="*50)
                print(f"Surmise PCGP: {meta['function']} Function Results".center(50))
                print("="*50)
                print(f"{'Training RMSE:':<20}{train_rmse:.4f}")
                print(emu)  
                print("="*50)
        
        plt.xlabel('First Input Dimension')
        plt.ylabel(f'Output (Dimension {output_dim + 1})')
        plt.title(f'PCGP vs Surmise Comparison for {meta["function"]} Function')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        print("\n" + "="*60)
        print(f"COMPARISON SUMMARY: {meta['function']} Function".center(60))
        print("="*60)
        print(f"{'PCGP RMSE:':<30}{results['pcgp']:.4f}")
        print(f"{'Surmise RMSE:':<30}{results['surmise']:.4f}")
        print(f"{'Difference:':<30}{abs(results['pcgp'] - results['surmise']):.4f}")
        print("="*60 + "\n")
        
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_experiments()
