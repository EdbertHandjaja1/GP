import numpy as np
import time
import sys
import os
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
    for n in ns:
        for function in test_functions:
            func_caller = TestFuncCaller(function)
            meta = func_caller.info
            
            # input train
            theta_train = np.uniform(0, 1, (n, meta['thetadim']))
            X_train = np.random.uniform(0, 1, (n, meta['xdim']))
            
            # output train
            Y_train = func_caller.info['nofailmodel'](X_train, theta_train)
            Y_true_train = func_caller.info['true_func'](X_train)
            Y_train += noise_level * np.std(Y_train)

            # test data
            X_test = np.random.uniform(0, 1, (ntest, meta['xdim']))
            theta_test = np.random.uniform(0, 1, (ntest, meta['thetadim']))
            Y_test_true = func_caller.info['nofailmodel'](X_test, theta_test)

            for output in output_dims:
                # start timer

                # run pcgp
                Y_pred_mean, Y_pred_std = run_pcgp(
                    n_components=1,
                    input_dim=meta['xdim'],
                    output_dim=output,
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test
                )

                # stop timer
                # calculate rmse

                # run surmise
                Y_pred_mean, Y_pred_std, emu = run_surmise(
                    n_components=1,
                    input_dim=meta['xdim'],
                    output_dim=output,
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test
                )

                # stop timer
                # calculate rmse

                pass

            # Store results
        
        # SAVE results to output 
    

if __name__ == "__main__":
    main()