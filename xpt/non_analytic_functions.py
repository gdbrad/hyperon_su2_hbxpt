import gvar as gv
import numpy as np
import scipy.special as ss

### non-analytic functions that arise in extrapolation formulae for hyperon masses

def fcn_L(m, mu):
    output = m**2 * np.log(m**2 / mu**2)
    return output

def fcn_L_bar(m,mu):
    output = m**4 * np.log(m**2 / mu**2)
    return output

def fcn_R(g):

#if isinstance(g, gv._gvarcore.GVar):
    x = g
    conds = [(x > 0) & (x <= 1), x > 1]
    funcs = [lambda x: np.sqrt(1-x) * np.log((1-np.sqrt(1-x))/(1+np.sqrt(1-x))),
                lambda x: 2*np.sqrt(x-1)*np.arctan(np.sqrt(x-1))
                ]

    pieces = np.piecewise(x, conds, funcs)
    return pieces

def fcn_dR(g):
#if isinstance(g, gv._gvarcore.GVar):
    x = g
    conds = [(x > 0) & (x < 1), x==1, x > 1]
    funcs = [lambda x: 1/x - np.log((1-np.sqrt(1-x))/(np.sqrt(1-x)+1))/(2*np.sqrt(1-x)),
                lambda x: x==2,
                lambda x: 1/x + np.arctan(np.sqrt(x-1)) / np.sqrt(x-1)
                ]

    pieces = np.piecewise(x, conds, funcs)
    return pieces


def fcn_F(eps_pi, eps_delta):
    output = (
        - eps_delta *(eps_delta**2 - eps_pi**2) *fcn_R((eps_pi/eps_delta)**2)
        - (3/2) *eps_pi**2 *eps_delta *np.log(eps_pi**2)
        - eps_delta**3 *np.log(4 *(eps_delta/eps_pi)**2)
    )
    return output

def fcn_dF(eps_pi, eps_delta):
    output = 0
    output += (
        + 2*eps_delta**3 / eps_pi
        - 3*eps_delta*eps_pi *np.log(eps_pi**2) 
        - 3*eps_delta*eps_pi
        + (2*eps_pi**3 / eps_delta - 2*eps_delta*eps_pi)*fcn_dR(eps_pi**2/eps_delta**2)
        + 2*eps_pi*eps_delta*fcn_R(eps_pi**2/eps_delta**2)
    )
    return output

def fcn_J(eps_pi, eps_delta):
    output = 0
    output += eps_pi**2 * np.log(eps_pi**2)
    output += 2*eps_delta**2 * np.log((4*eps_delta**2)/ eps_pi**2)
    output += 2*eps_delta**2 * fcn_R(eps_pi**2/eps_delta**2)

    return output

def fcn_dJ(eps_pi,eps_delta):
    output = 0
    output -= 4*eps_delta**2/eps_pi 
    output += 4*eps_pi*fcn_dR(eps_pi**2/eps_delta**2) 
    output += 2*eps_pi*np.log(eps_pi**2) + 2*eps_pi
    return output


    
