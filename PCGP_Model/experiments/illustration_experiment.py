import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from helper import run_pcgp, run_surmise
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator

def evaluate_coverage(intervals, true_values, is_ci=False):
    """Evaluates coverage probability of CIs or PIs."""
    lower, upper = intervals
    covered = np.logical_and(true_values >= lower, true_values <= upper)
    coverage = np.mean(covered) * 100
    print(f"{'CI' if is_ci else 'PI'} Coverage: {coverage:.2f}%")
    return coverage

def run_experiments(n_train=100, n_test=50, noise_level=0.05, output_dim_idx=0):
    """
    Runs experiments comparing PCGP and Surmise with CI/PI evaluation.
    """
    test_functions = ['borehole', 'otlcircuit', 'piston', 'wingweight']
    models = ['pcgp', 'surmise']
    
    for function_name in test_functions:
        func_caller = TestFuncCaller(function_name)
        meta = func_caller.info
        
        # generate training data once for both models
        theta_train = np.random.uniform(0, 1, (n_train, meta['thetadim'])) 
        X_train = np.random.uniform(0, 1, (n_train, meta['xdim']))

        Y_train = func_caller.info['nofailmodel'](X_train, theta_train)
        Y_true_train = func_caller.info['true_func'](X_train)  # f(x)
        
        Y_train += noise_level * np.std(Y_train) * np.random.randn(*Y_train.shape)  

        X_test = np.random.uniform(0, 1, (n_test, meta['xdim']))
        theta_test = np.random.uniform(0, 1, (n_test, meta['thetadim']))
        Y_test_true = func_caller.info['nofailmodel'](X_test, theta_test) # f(x)
        Y_test_noisy = Y_test_true + noise_level * np.std(Y_test_true) * np.random.randn(*Y_test_true.shape) # y(x)

        sort_idx_test = np.argsort(X_test[:, 0])

        plt.figure(figsize=(15, 8))
        
        plt.scatter(X_train[:, 0], Y_train[:, output_dim_idx], c='black', marker='x', 
                   s=100, label='Noisy Training Data', alpha=0.6)
        plt.plot(X_test[sort_idx_test, 0], Y_test_true[sort_idx_test, output_dim_idx], 
                'r-', linewidth=3, label='True Function (f(x))')

        results = {}
        
        for model in models:
            if model == 'pcgp':
                Y_pred_mean, Y_pred_std, _, _ = run_pcgp(
                    n_components=1,
                    input_dim=meta['xdim'],
                    output_dim_idx=output_dim_idx, 
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test
                )
                
                # confidence interval 
                ci_lower = Y_pred_mean - 1.96 * Y_pred_std  
                ci_upper = Y_pred_mean + 1.96 * Y_pred_std
                ci_coverage = evaluate_coverage(
                    (ci_lower[sort_idx_test, 0], ci_upper[sort_idx_test, 0]), 
                    Y_test_true[sort_idx_test, output_dim_idx], 
                    is_ci=True
                )

                # prediction interval 
                pi_lower = Y_pred_mean - 1.96 * np.sqrt(Y_pred_std**2 + noise_level**2)  
                pi_upper = Y_pred_mean + 1.96 * np.sqrt(Y_pred_std**2 + noise_level**2)
                pi_coverage = evaluate_coverage(
                    (pi_lower[sort_idx_test, 0], pi_upper[sort_idx_test, 0]), 
                    Y_test_noisy[sort_idx_test, output_dim_idx], 
                    is_ci=False
                )

                plt.plot(X_test[sort_idx_test, 0], Y_pred_mean[sort_idx_test, 0], 'b-', 
                        linewidth=2, label='PCGP Predicted Mean')
                plt.fill_between(X_test[sort_idx_test, 0], ci_lower[sort_idx_test, 0], ci_upper[sort_idx_test, 0],
                                color='blue', alpha=0.2, label='PCGP 95% CI (Mean)')
                plt.fill_between(X_test[sort_idx_test, 0], pi_lower[sort_idx_test, 0], pi_upper[sort_idx_test, 0],
                                color='cyan', alpha=0.1, label='PCGP 95% PI (Observation)')

                results['pcgp'] = {
                    'rmse': np.sqrt(np.mean((Y_pred_mean - Y_test_true[:, output_dim_idx])**2)),
                    'ci_coverage': ci_coverage,
                    'pi_coverage': pi_coverage
                }

            else:  # surmise
                Y_pred_mean, Y_pred_std, _, _, _ = run_surmise(
                    n_components=1,
                    input_dim=meta['xdim'],
                    output_dim_idx=output_dim_idx, 
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test
                )
                
                # confidence interval 
                ci_lower = Y_pred_mean - 1.96 * Y_pred_std
                ci_upper = Y_pred_mean + 1.96 * Y_pred_std
                ci_coverage = evaluate_coverage(
                    (ci_lower[sort_idx_test, 0], ci_upper[sort_idx_test, 0]), 
                    Y_test_true[sort_idx_test, output_dim_idx], 
                    is_ci=True
                )

                # prediction interval 
                pi_lower = Y_pred_mean - 1.96 * np.sqrt(Y_pred_std**2 + noise_level**2)
                pi_upper = Y_pred_mean + 1.96 * np.sqrt(Y_pred_std**2 + noise_level**2)
                pi_coverage = evaluate_coverage(
                    (pi_lower[sort_idx_test, 0], pi_upper[sort_idx_test, 0]), 
                    Y_test_noisy[sort_idx_test, output_dim_idx], 
                    is_ci=False
                )

                plt.plot(X_test[sort_idx_test, 0], Y_pred_mean[sort_idx_test, 0], 'g--', 
                        linewidth=2, label='Surmise Predicted Mean')
                plt.fill_between(X_test[sort_idx_test, 0], ci_lower[sort_idx_test, 0], ci_upper[sort_idx_test, 0],
                                color='green', alpha=0.15, label='Surmise 95% CI (Mean)')
                plt.fill_between(X_test[sort_idx_test, 0], pi_lower[sort_idx_test, 0], pi_upper[sort_idx_test, 0],
                                color='lime', alpha=0.05, label='Surmise 95% PI (Observation)')

                results['surmise'] = {
                    'rmse': np.sqrt(np.mean((Y_pred_mean - Y_test_true[:, output_dim_idx])**2)),
                    'ci_coverage': ci_coverage,
                    'pi_coverage': pi_coverage
                }

        plt.xlabel('First Input Dimension')
        plt.ylabel(f'Output (Dimension {output_dim_idx + 1})')
        plt.title(f'{meta["function"]} Function: PCGP vs Surmise (CIs/PIs)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        print("\n" + "="*70)
        print(f"COMPARISON SUMMARY: {meta['function']}".center(70))
        print("="*70)
        print(f"{'Metric':<20}{'PCGP':<25}{'Surmise':<25}")
        print("-"*70)
        print(f"{'RMSE':<20}{results['pcgp']['rmse']:.4f}{'':<25}{results['surmise']['rmse']:.4f}")
        print(f"{'CI Coverage (%)':<20}{results['pcgp']['ci_coverage']:.2f}{'':<25}{results['surmise']['ci_coverage']:.2f}")
        print(f"{'PI Coverage (%)':<20}{results['pcgp']['pi_coverage']:.2f}{'':<25}{results['surmise']['pi_coverage']:.2f}")
        print("="*70 + "\n")

        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_experiments()