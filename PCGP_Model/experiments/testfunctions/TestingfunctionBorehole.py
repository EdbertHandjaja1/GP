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
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    (Treffs, Hus, LdKw, powparams) = np.split(tstd, tstd.shape[1], axis=1)

    Treff = (0.5 - 0.05) * Treffs + 0.05
    Hu = Hus * (1110 - 990) + 990
    if hard:
        Ld_Kw = LdKw * (1680 / 1500 - 1120 / 15000) + 1120 / 15000
    else:
        Ld_Kw = LdKw * (1680 / 9855 - 1120 / 12045) + 1120 / 12045

    powparam = powparams * (0.5 - (-0.5)) + (-0.5)

    return np.hstack((Hu, Ld_Kw, Treff, powparam))

def xstd2x(xstd):
    if xstd.ndim < 1.5:
        xstd = xstd[:, None].T
    (rws, Hls) = np.split(xstd, xstd.shape[1], axis=1)

    rw = rws * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw)
    Hl = Hls * (820 - 700) + 700

    return np.hstack((rw, Hl))

def borehole_vec(x, theta):
    (Hu, Ld_Kw, Treff, powparam) = np.split(theta, theta.shape[1], axis=1)
    (rw, Hl) = np.split(x, x.shape[1], axis=1)
    numer = 2 * np.pi * (Hu - Hl)
    denom1 = 2 * Ld_Kw / rw ** 2
    denom2 = Treff
    return ((numer / ((denom1 + denom2))) * np.exp(powparam * rw)).reshape(-1)

def borehole_model(x, theta):
    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = borehole_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T

def borehole_true(x):
    theta0 = np.atleast_2d(np.array([0.5] * 4))
    return borehole_model(x, theta0)