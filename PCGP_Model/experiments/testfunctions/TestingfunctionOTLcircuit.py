import numpy as np

def query_func_meta():
    return {
        'function': 'OTLcircuit',
        'xdim': 4, 
        'thetadim': 2,
        'c_array': np.arange(0, 10, 0.01),
        'fail_array': np.array([]) 
    }

def tstd2theta_otl(tstd):
    if tstd.ndim == 1:
        tstd = tstd.reshape(1, -1)
    
    Rfs, betas = tstd[:, 0], tstd[:, 1]
    
    Rf = 0.5 + Rfs * (3 - 0.5) 
    beta = 50 + betas * (300 - 50) 
    
    return np.column_stack((Rf, beta))

def xstd2x_otl(xstd):
    if xstd.ndim == 1:
        xstd = xstd.reshape(1, -1)
    
    Rb1s, Rb2s, Rc1s, Rc2s = xstd[:, 0], xstd[:, 1], xstd[:, 2], xstd[:, 3]
    
    Rb1 = 50 + Rb1s * (150 - 50)  
    Rb2 = 25 + Rb2s * (75 - 25)   
    Rc1 = 1.2 + Rc1s * (2.5 - 1.2)  
    Rc2 = 0.25 + Rc2s * (1.2 - 0.25)  
    
    return np.column_stack((Rb1, Rb2, Rc1, Rc2))

def OTLcircuit_model(x, theta):
    theta_actual = tstd2theta_otl(theta)
    x_actual = xstd2x_otl(x)
    
    Rb1, Rb2, Rc1, Rc2 = x_actual[:, 0], x_actual[:, 1], x_actual[:, 2], x_actual[:, 3]
    Rf, beta = theta_actual[:, 0], theta_actual[:, 1]
    
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    term1 = (Vb1 + 0.74) * beta * (Rc2 + 9)
    term2 = beta * (Rc2 + 9) + Rf
    term3 = 11.35 * Rf
    term4 = beta * (Rc2 + 9) + Rf
    
    Vm = (term1 / term2) + (term3 / term4)
    
    return Vm.reshape(-1, 1)

def OTLcircuit_true(x):
    n = x.shape[0]
    theta0 = np.tile([0.5, 0.5], (n, 1))  
    result = OTLcircuit_model(x, theta0)
    return result