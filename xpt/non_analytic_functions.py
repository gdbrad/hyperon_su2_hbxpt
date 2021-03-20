import gvar as gv
import numpy as np
import scipy.special as ss
import i_o

LECs = i_o.InputOutput.get_data_phys_point('lam_chi')
print(LECs)
lam_chi = LECs['lam_chi']
m_pi = LECs['m_pi']
m_k = LECs['m_k']
m_xi = LECs['m_xi']
m_xi_st = LECs['m_xi_st']
print(m_k)
eps_pi = gv.mean(m_pi) / gv.mean(lam_chi) # set to zero in the chiral limit 

# for Xi system, delta is mass splitting between Xi* and Xi
delta = gv.mean(m_xi_st) - gv.mean(m_xi)
print(delta)

#eps_pi = gv.mean(m_pi) / gv.mean(lam_chi) # set to zero in the chiral limit 
eps_delta = delta / gv.mean(lam_chi)

### non-analytic functions that arise in extrapolation formulae for hyperon masses

def fcn_R(g):
    if isinstance(g, gv._gvarcore.GVar):
        x = gv.mean(g)
        #print(x)

        if (x>0 and x<=1):
        #a = np.sqrt(1-x)
            R = np.sqrt(1-x) * np.log((1-np.sqrt(1-x))/(1+np.sqrt(1-x)))
        elif x>1:
        #b = np.sqrt(x-1)
            R = 2*np.sqrt(x-1)*np.arctan(np.sqrt(x-1))
    #not gvar
    else:
        if (g>0 and g<=1):
        #a = np.sqrt(1-x)
            R = np.sqrt(1-g) * np.log((1-np.sqrt(1-g))/(1+np.sqrt(1-g)))
        elif x>1:
        #b = np.sqrt(x-1)
            R = 2*np.sqrt(g-1)*np.arctan(np.sqrt(g-1))


    return R

eps_pi_delta = eps_pi**2 / eps_delta**2
#fcn_R(eps_pi_delta)


def fcn_F(eps_pi, eps_delta):

    output = -1*eps_delta*(eps_delta**2 - eps_pi**2) * fcn_R(eps_pi**2 / eps_delta**2)
    output = output - ((3/2) * eps_pi**2 * eps_delta * np.log(eps_pi**2))
    output = output - (eps_delta**3 * np.log((4*eps_delta**2) / (eps_pi**2)))
    
    return output


def fcn_J(eps_pi, eps_delta):
    out = eps_pi**2 * np.log(eps_pi**2)
    out = out +  2*eps_delta**2 * np.log((4*eps_delta**2)/ eps_pi**2)
    out = out +  2*eps_delta**2 * fcn_R(eps_pi**2/eps_delta**2)

    return out


    
