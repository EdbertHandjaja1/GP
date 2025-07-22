import numpy as np

def query_func_meta():
    return {
        'function': 'OTLcircuit',
        'xdim': 6,
        'thetadim': 0,  
    }

def OTLcircuit_model(x, theta=None):
    """OTL Circuit function"""
    if theta is not None:
        x = np.hstack((x, theta))
    
    Rb1 = 50 * x[:, 0] + 150
    Rb2 = 25 * x[:, 1] + 125
    Rf = 0.5 * x[:, 2] + 4.5
    Rc1 = 1.2 * x[:, 3] + 2.8
    Rc2 = 0.25 * x[:, 4] + 0.75
    beta = 50 * x[:, 5] + 150
    
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    term1 = (Vb1 + 0.74) * beta * (Rc2 + 9)
    term2 = beta * (Rc2 + 9) + Rf
    term3 = 11.35 * Rf
    term4 = beta * (Rc2 + 9) + Rf
    
    f = (term1 / term2) + (term3 / term4)
    return f.reshape(-1, 1)

def OTLcircuit_true(x):
    return OTLcircuit_model(x)