import numpy as np
import time
import sys
import os
import pandas as pd
from joblib import Parallel, delayed
import pathlib
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator
from helper import run_pcgp, run_surmise

outputdir = r'experiments/output/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

output_dims = [1]
ns = [1000, 3000, 5000]
test_functions = ['borehole', 'otlcircuit', 'piston']
ntest = 150
noise_level = 0.05
n_reps = 1

def calculate_rmse(ytrue, ypred):
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse

def evaluate_coverage(ytrue, ypred_mean, ypred_std, noise_level=0.05):
    """Evaluates coverage probability of prediction intervals."""
    # confidence interval coverage 
    ci_lower = ypred_mean - 1.96 * ypred_std
    ci_upper = ypred_mean + 1.96 * ypred_std
    ci_covered = np.logical_and(ytrue >= ci_lower, ytrue <= ci_upper)
    ci_coverage = np.mean(ci_covered) * 100
    
    # prediction interval coverage 
    pi_lower = ypred_mean - 1.96 * np.sqrt(ypred_std**2 + noise_level**2)
    pi_upper = ypred_mean + 1.96 * np.sqrt(ypred_std**2 + noise_level**2)
    pi_covered = np.logical_and(ytrue >= pi_lower, ytrue <= pi_upper)
    pi_coverage = np.mean(pi_covered) * 100
    
    return ci_coverage, pi_coverage

def run_experiment(n, function, output_idx_to_model, rep):
    """
    Function to run a single experiment (for a given n, function, output dimension, and repetition).
    This function will be parallelized.
    """
    # use different seed for each repetition to get different data
    np.random.seed(42 + 1000 * rep)  
    
    func_caller = TestFuncCaller(function)
    meta = func_caller.info

    # input train
    theta_train = np.random.uniform(0, 1, (n, meta['thetadim']))
    X_train = np.random.uniform(0, 1, (n, meta['xdim']))

    # output train
    Y_train = func_caller.info['nofailmodel'](X_train, theta_train)
    Y_train += noise_level * np.std(Y_train, axis=0, keepdims=True) * np.random.normal(0, 1, Y_train.shape)

    # test
    X_test = np.random.uniform(0, 1, (ntest, meta['xdim']))
    theta_test = np.random.uniform(0, 1, (ntest, meta['thetadim']))
    Y_test_true = func_caller.info['nofailmodel'](X_test, theta_test)
    Y_test_noisy = Y_test_true + noise_level * np.std(Y_test_true, axis=0, keepdims=True) * np.random.normal(0, 1, Y_test_true.shape)

    results = []

    # PCGP
    Y_pred_mean_pcgp, Y_pred_std_pcgp, train_time, pred_time = run_pcgp(
        n_components=1,
        input_dim=meta['xdim'],
        output_dim_idx=output_idx_to_model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test
    )

    rmse_pcgp = calculate_rmse(Y_test_true[:, output_idx_to_model].reshape(-1, 1), Y_pred_mean_pcgp)
    ci_coverage_pcgp, pi_coverage_pcgp = evaluate_coverage(
        Y_test_true[:, output_idx_to_model].reshape(-1, 1),
        Y_pred_mean_pcgp,
        Y_pred_std_pcgp,
        noise_level
    )

    results.append({
        'model': 'PCGP',
        'n_train': n,
        'function': function,
        'output_dim_modeled_idx': output_idx_to_model,
        'rep': rep,
        'training_time': train_time,
        'prediction_time': pred_time,
        'rmse': rmse_pcgp,
        'ci_coverage': ci_coverage_pcgp,
        'pi_coverage': pi_coverage_pcgp
    })

    # surmise
    Y_pred_mean_surmise, Y_pred_std_surmise, emu, train_time, pred_time = run_surmise(
        n_components=1,
        input_dim=meta['xdim'],
        output_dim_idx=output_idx_to_model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test
    )
    rmse_surmise = calculate_rmse(Y_test_true[:, output_idx_to_model].reshape(-1, 1), Y_pred_mean_surmise)
    ci_coverage_surmise, pi_coverage_surmise = evaluate_coverage(
        Y_test_true[:, output_idx_to_model].reshape(-1, 1),
        Y_pred_mean_surmise,
        Y_pred_std_surmise,
        noise_level
    )

    results.append({
        'model': 'Surmise',
        'n_train': n,
        'function': function,
        'output_dim_modeled_idx': output_idx_to_model,
        'rep': rep,
        'training_time': train_time,
        'prediction_time': pred_time,
        'rmse': rmse_surmise,
        'ci_coverage': ci_coverage_surmise,
        'pi_coverage': pi_coverage_surmise
    })
    
    return results

def main():
    all_results = []
    
    tasks = []
    for rep in range(n_reps):
        for n in ns:
            for function in test_functions:
                for output_idx_to_model in [0]:
                    tasks.append((n, function, output_idx_to_model, rep))
    
    parallel_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_experiment)(n, function, output_idx_to_model, rep)
        for n, function, output_idx_to_model, rep in tasks
    )

    for res_list in parallel_results:
        all_results.extend(res_list)
    
    results_df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'numerical_results_{timestamp}.csv'
    filepath = os.path.join(outputdir, filename)
    
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    
    return filepath


if __name__ == "__main__":
    main()