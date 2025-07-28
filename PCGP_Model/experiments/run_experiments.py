
import numpy as np
import time
import sys
import os
import pandas as pd
from joblib import Parallel, delayed
import pathlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator
from helper import run_pcgp, run_surmise

outputdir = r'experiments/output/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

output_dims = [1]
# ns = [100, 200, 300]
# cap at 1000
ns = [500]
test_functions = ['borehole', 'otlcircuit', 'piston']
ntest = 150
noise_level = 0.05

def calculate_rmse(ytrue, ypred):
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse

def run_experiment(n, function, output_idx_to_model):
    """
    Function to run a single experiment (for a given n, function, and output dimension).
    This function will be parallelized.
    """
    np.random.seed(42) 
    
    func_caller = TestFuncCaller(function)
    meta = func_caller.info

    # input train
    theta_train = np.random.uniform(0, 1, (n, meta['thetadim']))
    X_train = np.random.uniform(0, 1, (n, meta['xdim']))

    # output train
    Y_train = func_caller.info['nofailmodel'](X_train, theta_train)
    Y_train += noise_level * np.std(Y_train)

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

    results.append({
        'model': 'PCGP',
        'n_train': n,
        'function': function,
        'output_dim_modeled_idx': output_idx_to_model,
        'training_time': train_time,
        'prediction_time': pred_time,
        'rmse': rmse_pcgp
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

    results.append({
        'model': 'Surmise',
        'n_train': n,
        'function': function,
        'output_dim_modeled_idx': output_idx_to_model,
        'training_time': train_time,
        'prediction_time': pred_time,
        'rmse': rmse_surmise
    })

    return results

def main():
    all_results = []
    
    tasks = []
    for n in ns:
        for function in test_functions:
            for output_idx_to_model in [0]:
                tasks.append((n, function, output_idx_to_model))

    parallel_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_experiment)(n, function, output_idx_to_model)
        for n, function, output_idx_to_model in tasks
    )

    for res_list in parallel_results:
        all_results.extend(res_list)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(outputdir, 'numerical_results.csv'), index=False)
    print(f"Results saved to {os.path.join(outputdir, 'numerical_results.csv')}")

if __name__ == "__main__":
    main()