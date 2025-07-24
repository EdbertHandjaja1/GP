import numpy as np
import time
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator
import pathlib
from helper import run_pcgp, run_surmise

outputdir = r'experiments/output/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

output_dims = [1]
ns = [100, 200, 300]  
test_functions = ['borehole', 'otlcircuit', 'piston']
ntest = 150

def calculate_rmse(ytrue, ypred):
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse

def main():
    np.random.seed(42)
    noise_level = 0.05
    all_results = [] 

    for n in ns:
        for function in test_functions:
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

            for output_idx_to_model in [0]: 
                # PCGP
                start_time_pcgp = time.time()
                Y_pred_mean_pcgp, Y_pred_std_pcgp = run_pcgp(
                    n_components=1,
                    input_dim=meta['xdim'],
                    output_dim_idx=output_idx_to_model,
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test
                )
                end_time_pcgp = time.time()
                time_pcgp = end_time_pcgp - start_time_pcgp

                rmse_pcgp = calculate_rmse(Y_test_true[:, output_idx_to_model].reshape(-1, 1), Y_pred_mean_pcgp)

                all_results.append({
                    'model': 'PCGP',
                    'n_train': n,
                    'function': function,
                    'output_dim_modeled_idx': output_idx_to_model, 
                    'training_time': time_pcgp,
                    'rmse': rmse_pcgp
                })

                # surmise
                start_time_surmise = time.time()
                Y_pred_mean_surmise, Y_pred_std_surmise, emu = run_surmise(
                    n_components=1, 
                    input_dim=meta['xdim'],
                    output_dim_idx=output_idx_to_model, 
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test
                )
                end_time_surmise = time.time()
                time_surmise = end_time_surmise - start_time_surmise
                rmse_surmise = calculate_rmse(Y_test_true[:, output_idx_to_model].reshape(-1, 1), Y_pred_mean_surmise)

                all_results.append({
                    'model': 'Surmise',
                    'n_train': n,
                    'function': function,
                    'output_dim_modeled_idx': output_idx_to_model, 
                    'training_time': time_surmise,
                    'rmse': rmse_surmise
                })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(outputdir, 'numerical_results.csv'), index=False)
    print(f"Results saved to {os.path.join(outputdir, 'numerical_results.csv')}")

if __name__ == "__main__":
    main()