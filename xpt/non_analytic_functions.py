import gvar as gv
import numpy as np
import scipy.special as ss
import i_o

### non-analytic functions that arise in extrapolation formulae for hyperon masses


def fcn_R(g):

#if isinstance(g, gv._gvarcore.GVar):
    x = gv.mean(g)
    conds = [(x > 0) & (x <= 1), x > 1]
    funcs = [lambda x: np.sqrt(1-x) * np.log((1-np.sqrt(1-x))/(1+np.sqrt(1-x))),
                lambda x: 2*np.sqrt(x-1)*np.arctan(np.sqrt(x-1))
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


def fcn_J(eps_pi, eps_delta):
    out = eps_pi**2 * np.log(eps_pi**2)
    out = out +  2*eps_delta**2 * np.log((4*eps_delta**2)/ eps_pi**2)
    out = out +  2*eps_delta**2 * fcn_R(eps_pi**2/eps_delta**2)

    return out


    
