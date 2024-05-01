import numpy as np
def localizer_poly_2d(x, r1, d):
    x = np.asarray(x)
    x = x[:, :2] / r1
    r = np.linalg.norm(x, axis=1) if x.ndim > 1 else np.linalg.norm(x)
    val = (1 - r ** d) ** d
    if x.ndim > 1:
        val[r > 1] = 0
    elif r > 1:
        val = 0
    return val
def localizer_exp_2d(x, r1, d):
    c = 1.0
    x = np.asarray(x)
    x = x[:, :2] / r1
    r = np.linalg.norm(x, axis=1) if x.ndim > 1 else np.linalg.norm(x)
    val = np.exp(-c/(1.-np.power(r,d))) / np.exp(-c)
    if x.ndim > 1:
        val[r > 1] = 0
    elif r > 1:
        val = 0
    return val

def localizer_int_exp_2d(x, r1, d):
    x = np.asarray(x)
    x = x[:, :2] / r1
    r = np.linalg.norm(x, axis=1) if x.ndim > 1 else np.linalg.norm(x)

    c = 1.0
    g = lambda x: np.exp(-c/(1-x**d))

    from scipy.integrate import quad
    int_g = quad(g,-1,1)
    h = lambda x: quad(g,-1,x) / int_g
    f = lambda y: h(1-2*np.abs(1))

    val = f(r)

    if x.ndim > 1:
        val[r > 1] = 0
    elif r > 1:
        val = 0
    return val