import numpy as np

def query_func_meta():
    return {
        'function': 'Piston',
        'xdim': 7,
        'thetadim': 0,  
    }

def Piston_model(x, theta=None):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    M = 30 * x[:, 0] + 30      
    S = 0.015 * x[:, 1] + 0.005
    V0 = 0.008 * x[:, 2] + 0.002
    k = 3000 * x[:, 3] + 1000   
    P0 = 90000 * x[:, 4] + 10000
    Ta = 290 * x[:, 5] + 10     
    T0 = 340 * x[:, 6] + 10     
    
    A = P0 * S + 19.62 * M - (k * V0 / S)
    V = (S / (2 * k)) * (np.sqrt(A**2 + 4 * k * P0 * V0 / T0 * Ta) - A)
    C = 2 * np.pi * np.sqrt(M / (k + S**2 * P0 * V0 / T0 * Ta / V**2))
    
    return C.reshape(-1, 1)

def Piston_true(x):
    return Piston_model(x)