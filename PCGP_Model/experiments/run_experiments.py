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

rep_n = 5 
output_dims = [1]
ns = [50, 100, 200]  
funcs = ['borehole', 'otlcircuit', 'piston', 'wingweight']  
ntest = 150

def calculate_rmse(ytrue, ypred):
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse

def run_pcgp_model(data, output_dim, n_components=None):
    xtrain, xtest, ytrain = data['xtrain'], data['xtest'], data['ytrain']
    
    if n_components is None:
        n_components = min(output_dim, 10)
    
    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=xtrain.shape[1],
        output_dim=output_dim
    )
    
    ranges = np.column_stack([np.zeros(xtrain.shape[1]), np.ones(xtrain.shape[1])])
    
    fitted_model = pcgp.fit(xtrain, ytrain.T, ranges)
    
    pred_mean, _ = fitted_model.predict(xtest, ranges, return_std=True)
    
    return pred_mean.T

def run_surmise_pcgp_model(data, output_dim):
    xtrain, xtest, ytrain = data['xtrain'], data['xtest'], data['ytrain']
    
    theta_emu_train = xtrain

    x_emu_train = np.arange(output_dim).reshape(-1, 1)  
    
    f_emu_train = ytrain  
    
    emu = emulator(
        x=x_emu_train,       
        theta=theta_emu_train, 
        f=f_emu_train,    
        method='PCGP',
        options={'epsilon': 0}
    )
    emu.fit()
    
    pred = emu.predict(x=x_emu_train, theta=xtest)
    pred_mean = pred.mean().T  
    
    return pred_mean

def main():
    np.random.seed(42)
    
    pass

if __name__ == "__main__":
    main()