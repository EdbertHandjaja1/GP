import numpy as np

def query_func_meta():
    return {
        'function': 'Borehole',
        'xdim': 2,
        'thetadim': 4,
        'c_array': np.arange(0, 10, 0.01),
        'fail_array': np.array([...]) 
    }

def tstd2theta(tstd, hard=True):
    if tstd.ndim == 1:
        tstd = tstd.reshape(1, -1)
    
    Treffs, Hus, LdKw, powparams = tstd[:, 0], tstd[:, 1], tstd[:, 2], tstd[:, 3]

    Treff = (0.5 - 0.05) * Treffs + 0.05  
    Hu = Hus * (1110 - 990) + 990      
    
    if hard:
        Ld_Kw = LdKw * (1680 / 1500 - 1120 / 15000) + 1120 / 15000
    else:
        Ld_Kw = LdKw * (1680 / 9855 - 1120 / 12045) + 1120 / 12045

    powparam = powparams * (0.5 - (-0.5)) + (-0.5)  

    return np.column_stack((Hu, Ld_Kw, Treff, powparam))

def xstd2x(xstd):
    if xstd.ndim == 1:
        xstd = xstd.reshape(1, -1)
    
    rws, Hls = xstd[:, 0], xstd[:, 1]

    rw = rws * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw) 
    Hl = Hls * (820 - 700) + 700 

    return np.column_stack((rw, Hl))

def borehole_model(x, theta):
    theta_actual = tstd2theta(theta)
    x_actual = xstd2x(x)
    
    n = x_actual.shape[0]
    
    rw = x_actual[:, 0]    
    Hl = x_actual[:, 1]   
    Hu = theta_actual[:, 0]     
    Ld_Kw = theta_actual[:, 1]   
    Treff = theta_actual[:, 2]  
    powparam = theta_actual[:, 3] 
    
    numer = 2 * np.pi * (Hu - Hl)
    denom1 = 2 * Ld_Kw / (rw ** 2)
    denom2 = Treff
    
    flow_rate = (numer / (denom1 + denom2)) * np.exp(powparam * rw)
    
    return flow_rate.reshape(-1, 1)

def borehole_true(x):
    """True borehole function with theta = [0.5, 0.5, 0.5, 0.5]"""
    n = x.shape[0]
    theta0 = np.tile([0.5, 0.5, 0.5, 0.5], (n, 1))  
    result = borehole_model(x, theta0)
    return result
