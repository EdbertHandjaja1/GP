import numpy as np

def query_func_meta():
    return {
        'function': 'Wingweight',
        'xdim': 10,
        'thetadim': 0,  
    }

def Wingweight_model(x, theta=None):
    """Wing weight function"""
    if theta is not None:
        x = np.hstack((x, theta))
    
    Sw = 150 * x[:, 0] + 50
    Wfw = 20 * x[:, 1] + 10
    A = 6 * x[:, 2] + 4
    Lambda = np.deg2rad(10 * x[:, 3] - 10)
    q = 16 * x[:, 4] - 8
    l = 0.5 * x[:, 5] + 0.5
    tc = 0.1 * x[:, 6] + 0.08
    Nz = 2.5 * x[:, 7] + 1.5
    Wdg = 2000 * x[:, 8] + 1000
    Wp = 1000 * x[:, 9] + 200
    
    term1 = 0.036 * Sw**0.758 * Wfw**0.0035
    term2 = (A / np.cos(Lambda)**2)**0.6
    term3 = q**0.006 * l**0.04
    term4 = (100 * tc / np.cos(Lambda))**(-0.3)
    term5 = (Nz * Wdg)**0.49
    
    W = term1 * term2 * term3 * term4 * term5 + Sw * Wp
    return W.reshape(-1, 1)

def Wingweight_true(x):
    return Wingweight_model(x)