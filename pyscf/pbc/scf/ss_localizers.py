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
    val = np.exp(-c / (1. - np.power(r, d))) / np.exp(-c)
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
    g = lambda x: np.exp(-c / (1 - x ** d))

    # To prevent error in h's integration bounds
    eps = np.finfo(float).eps
    r[r >= 1.0] = 1.0-eps

    from scipy.integrate import quad
    int_g = quad(g, -1+eps, 1-eps)[0]



    h = lambda x: quad(g, -1+eps, x,points=[-1,1])[0] / int_g if (x != -1) else 0
    f = lambda y: h(1 - 2 * np.abs(y))
    print(max(r),min(r))
    val = [f(ri) for ri in r]

    # if x.ndim > 1:
    #     val[r > 1] = 0
    # elif r > 1:
    #     val = 0
    return val

def localizer_rootexp_2d(x,r1,p=1.0,k=1.0):
    c = 1.0
    x = np.asarray(x)
    x = x[:, :2] / r1
    r = np.linalg.norm(x, axis=1) if x.ndim > 1 else np.linalg.norm(x)
    norm_fac = (p+1.5)/0.5
    val = np.exp(r**(2*k)/(r**2-1))*np.sin((p+1.5)*r)/np.sin(0.5*r) / norm_fac

    if x.ndim > 1:
        val[r > 1] = 0
        val[r == 0] = 1.0
        val[r == 1] = 0.0
    elif r > 1:
        val = 0
    return val