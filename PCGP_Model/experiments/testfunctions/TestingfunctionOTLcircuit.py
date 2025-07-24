import numpy as np

def query_func_meta():
    return {
        'function': 'OTLcircuit',
        'xdim': 4,  # Rb1, Rb2, Rc1, Rc2
        'thetadim': 2,  # Rf, beta
        'c_array': np.arange(0, 10, 0.01),
        'fail_array': np.array([])  # No failure cases by default
    }

def tstd2theta(tstd):
    """Convert standardized theta in [0,1] to physical parameters"""
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    (Rfs, betas) = np.split(tstd, tstd.shape[1], axis=1)
    
    Rf = 0.5 + Rfs * (3 - 0.5)  # Rf ∈ [0.5, 3]
    beta = 50 + betas * (300 - 50)  # beta ∈ [50, 300]
    
    return np.hstack((Rf, beta))

def xstd2x(xstd):
    """Convert standardized x in [0,1] to physical parameters"""
    if xstd.ndim < 1.5:
        xstd = xstd[:, None].T
    (Rb1s, Rb2s, Rc1s, Rc2s) = np.split(xstd, xstd.shape[1], axis=1)
    
    Rb1 = 50 + Rb1s * (150 - 50)  # Rb1 ∈ [50, 150]
    Rb2 = 25 + Rb2s * (75 - 25)   # Rb2 ∈ [25, 75]
    Rc1 = 1.2 + Rc1s * (2.5 - 1.2)  # Rc1 ∈ [1.2, 2.5]
    Rc2 = 0.25 + Rc2s * (1.2 - 0.25)  # Rc2 ∈ [0.25, 1.2]
    
    return np.hstack((Rb1, Rb2, Rc1, Rc2))

def OTLcircuit_vec(x, theta):
    """Vectorized computation of OTL circuit voltage"""
    (Rb1, Rb2, Rc1, Rc2) = np.split(x, x.shape[1], axis=1)
    (Rf, beta) = np.split(theta, theta.shape[1], axis=1)
    
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    term1 = (Vb1 + 0.74) * beta * (Rc2 + 9)
    term2 = beta * (Rc2 + 9) + Rf
    term3 = 11.35 * Rf
    term4 = beta * (Rc2 + 9) + Rf
    
    Vm = (term1 / term2) + (term3 / term4)
    return Vm.reshape(-1)

def OTLcircuit_model(x, theta):
    """Main model function matching LCGP interface"""
    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = OTLcircuit_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T

def OTLcircuit_true(x):
    """True function at default theta = [0.5, 0.5]"""
    theta0 = np.atleast_2d(np.array([0.5] * 2))
    return OTLcircuit_model(x, theta0)