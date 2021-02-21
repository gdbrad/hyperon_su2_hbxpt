import gvar as gv
import numpy as np
import scipy.special as ss


# define lecs 

lam_chi = 4*np.pi*F_pi
eps_pi
eps_delta 
### non-analytic functions that arise in extrapolation formulae for hyperon masses
def fcn_R(x):
    
    if (x>0 & x<=1):
        #a = np.sqrt(1-x)
        R = np.sqrt(1-x) * np.log((1-np.sqrt(1-x))/(1+np.sqrt(1-x)))
    elif x>1:
        #b = np.sqrt(x-1)
        R = 2*np.sqrt(x-1)*np.arctan(np.sqrt(x-1))
    return R


def fcn_F(eps_pi, eps_delta, mu):
    output = -eps_pi*(eps_delta**2 - eps_pi**2) * fcn_R(x=eps_pi**2/eps_delta**2) -
    ((3/2) * eps_pi**2 * eps_delta * np.log(eps_pi**2 * (lam_chi**2)/(mu**2) - eps_delta**3 *
    np.log((4*eps_delta**2)/ eps_pi**2) ))
    
    return output


def fcn_J(eps_pi, eps_delta, mu):
    output = eps_pi**2 * np.log(eps_pi**2 * (lam_chi**2)/(mu**2) + 2*eps_delta**2 * np.log((4*eps_delta**2)/ eps_pi**2) ) +
    2*eps_delta**2 * fcn_R(x=eps_pi**2/eps_delta**2)

    return output


    
