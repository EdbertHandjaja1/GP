
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
ns = [100, 200, 300, 500, 800]
test_functions = ['borehole', 'otlcircuit', 'piston']
ntest = 150
noise_level = 0.05
n_reps = 10 

def calculate_rmse(ytrue, ypred):
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse


def run_experiment(n, function, output_idx_to_model, rep):
    """
    Function to run a single experiment (for a given n, function, output dimension, and repetition).
    This function will be parallelized.
    """
    # use different seed for each repetition to get different data
    np.random.seed(42 + rep)  
    
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

    coverage_pcgp = 0

    results.append({
        'model': 'PCGP',
        'n_train': n,
        'function': function,
        'output_dim_modeled_idx': output_idx_to_model,
        'rep': rep,
        'training_time': train_time,
        'prediction_time': pred_time,
        'rmse': rmse_pcgp,
        'coverage': coverage_pcgp
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
    coverage_surmise = 0

    results.append({
        'model': 'Surmise',
        'n_train': n,
        'function': function,
        'output_dim_modeled_idx': output_idx_to_model,
        'rep': rep,
        'training_time': train_time,
        'prediction_time': pred_time,
        'rmse': rmse_surmise,
        'coverage': coverage_surmise
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

    print(f"Running {len(tasks)} total experiments...")
    
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
    
    summary_stats = results_df.groupby(['model', 'n_train', 'function']).agg({
        'rmse': ['mean', 'std', 'count'],
        'coverage': ['mean', 'std'],
        'training_time': ['mean', 'std'],
        'prediction_time': ['mean', 'std']
    }).round(4)
    
    summary_filename = f'summary_stats_{timestamp}.csv'
    summary_filepath = os.path.join(outputdir, summary_filename)
    summary_stats.to_csv(summary_filepath)
    print(f"Summary statistics saved to {summary_filepath}")
    
    return filepath

if __name__ == "__main__":
    main()