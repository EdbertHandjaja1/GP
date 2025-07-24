import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator
from pyDOE import lhs
import scipy.stats as sps
import pandas as pd
import pathlib

outputdir = r'experiments/output/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

output_dims = [1]
ns = [100, 200, 300]  
test_functions = ['borehole', 'otlcircuit', 'piston']
ntest = 150

def calculate_rmse(ytrue, ypred):
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse

def run_pcgp_model(data, output_dim, n_components=None):
    X_train, X_test, Y_train = data['xtrain'], data['xtest'], data['ytrain']
    
    if n_components is None:
        n_components = min(output_dim, 10)
    
    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=X_train.shape[1],
        output_dim=output_dim
    )
    
    ranges = np.column_stack([np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])])
    
    fitted_model = pcgp.fit(
        X_train, 
        Y_train[:, output_dim].reshape(-1, 1), 
        ranges
    )
    
    pred_mean, _ = fitted_model.predict(X_test, ranges, return_std=True)
    
    return pred_mean.T

def run_surmise_pcgp_model(data, output_dim):
    X_train, X_test, Y_train = data['xtrain'], data['xtest'], data['ytrain']
    
    theta_emu_train = X_train

    x_emu_train = np.array([[0]])
    
    f_emu_train = Y_train[:, output_dim].reshape(1, -1) 
    
    emu = emulator(
        x=x_emu_train,       
        theta=theta_emu_train, 
        f=f_emu_train,    
        method='PCGP',
        options={'epsilon': 0}
    )
    emu.fit()
    
    pred = emu.predict(x=x_emu_train, theta=X_test)
    pred_mean = pred.mean().T  
    
    return pred_mean

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
                pcgp = PrincipalComponentGaussianProcessModel(
                    n_components = 1,
                    input_dim=meta['xdim'],
                    output_dim=1
                )

                ranges = np.column_stack([np.zeros(meta['xdim']), np.ones(meta['xdim'])])

                fitted

                pass
        
    

if __name__ == "__main__":
    main()