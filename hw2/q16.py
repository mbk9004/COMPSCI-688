from scipy.optimize import minimize
import numpy as np

def rosen(d):
    return (1-d[0])**2 + 50*(d[1]-d[0]**2)**2

def drosen(d):
    return [ 2.0*(1-d[0])+200*d[0]*(d[1]-d[0]**2), -100.0*(d[1]-d[0]**2)]




res = minimize(rosen, [-1.0,1.0], jac=drosen,method='BFGS', options={'disp': True})
print(rosen(res.x))
print(res.x)