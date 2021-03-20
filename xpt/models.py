import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
import non_analytic_functions as naf
import i_o

class Xi(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(model, self).__init__(datatag)

        #override build data and build prior methods 
        # two models, models need to know part of data to each model use datatag
        self.model_info = model_info
        

    #fit_data from i_o module
    def fitfcn(self, p, fit_data=None, latt_spacing=None, observable=None):
        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]

    
        xi_a = (p['mpi'] / p['lam_chi']) #eps_pi
        xi_b = (p['m_delta'] / p['lam_chi']) #eps_delta
        xi_c = 1/2 * p['a/w'] #eps_a
    
        # p['s_{xi}'] = (2 *p['mk']**2 - p['mpi']**2) / p['lam_chi']**2
        # p['s_{xi,bar}'] = p['s_{xi}']**2 - 

        if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']:
            output  = self.fitfcn_lo_xpt(p) 
                    + self.fitfcn_nlo_xpt(p)
                    + self.fitfcn_n2lo_xpt(p)
            

        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:


        if self.model_info['order_ct'] in ['lo', 'nlo', 'n2lo']:
            output  = self.fitfcn_lo_ct(p)
                    + self.fitfcn_n2lo_ct(p)


        if self.model_info['xpt'] is True:
            output = (
                + p['m_{xi,0}'] #not-even leading order
                + self.fitfcn_lo_ct(p)
                + self.fitfcn_lo_xpt(p)
                + self.fitfcn_nlo_xpt(p)
                + self.fitfcn_n2lo_ct(p)
                + self.fitfcn_n2lo_xpt(p)
            )

        #print('good')
        return output

    

    def fitfcn_lo_ct(self, p):
        
        output = p['m_{xi,0}'] * (p['d_{xi,a}'] * (0.5* (p['a/w'])**2))

        return output

    def fitfcn_lo_xpt(self, p):
        output = p['{s_{xi}'] * p['lam_chi'] * xi_a

        return output

    def fitfcn_nlo_xpt(self,p):
        output = (
            ((3*np.pi * p['g_{xi,xi}']**2) / 2) 
            * p['lam_chi'] * p['a']**3 
            - (p['g_{xi_st,xi']**2 * p['lam_chi'] 
            * naf.fcn_F(xi_a, xi_b))
        )
        return output


    def fitfcn_n2lo_ct(self, p):     
        output = ( 
            + p['d_al'] * xi_a**2 * xi_c**2
            + p['d_aa'] * xi_c**2
            + p['b_{xi,4}'] *p['m_pi']**4 / p['lam_chi']**3 ##excluding ln(eps_pi^2)
        )

        return output

    def fitfcn_n2lo_xpt(self,p):
        output = (
            p['lam_chi'] * xi_a**2 * naf.fcn_J(xi_a, xi_b)
        )
        return output

    def buildprior(self, prior):
        return prior


    def builddata(self, data):
        return data[self.datatag]


class Xi_st(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(model, self).__init__(datatag)

        #override build data and build prior methods 
        # two models, models need to know part of data to each model use datatag
        self.model_info = model_info
        

    def fitfcn(self, p, fit_data=None):
        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]



        # Variables
        if xi is None:
            xi = {}
        if 'l' not in xi:
            xi['l'] = (p['mpi'] / p['lam_chi'])**2
        if 's' not in xi:
            xi['s'] = (2 *p['mk']**2 - p['mpi']**2) / p['lam_chi']**2

        y_ch = self.fitfcn_lo_ct(p, xi, latt_spacing)

        if 'a' not in xi:
            if self.observable == 'w0':
                xi['a'] =  1 / (2 *y_ch)**2
            elif self.observable == 't0':
                xi['a'] =  1 / (4 *y_ch)

        # lo
        #output = w0ch_a

        # nlo
        #output += w0ch_a *self.fitfcn_nlo_ct(p, xi)

        # n2lo
        #output += w0ch_a *self.fitfcn_n2lo_ct(p, xi)
        #output += w0ch_a *self.fitfcn_n2lo_log(p, xi)

        #return output

        #print(np.sqrt(xi['a'] ))

        #if isinstance(w0ch_a[0], gv._gvarcore.GVar):
        #    print(gv.evalcorr([w0ch_a[0], self.fitfcn_lo_ct(p, latt_spacing)[0]]))

        #output = 1 / (2 *np.sqrt(xi['a'])) *(
        output = y_ch *(
            + 1
            + self.fitfcn_nlo_ct(p, xi)
            + self.fitfcn_n2lo_ct(p, xi)
            + self.fitfcn_n2lo_log(p, xi)
        )

        #print('good')
        return output


    def fitfcn_lo_ct(self, p, xi, latt_spacing=None):

        if latt_spacing == 'a06':
            output = p['c0a06']
        elif latt_spacing == 'a09':
            output= p['c0a09']
        elif latt_spacing == 'a12':
            output = p['c0a12']
        elif latt_spacing == 'a15':
            output = p['c0a15']

        else:
            output = xi['l'] *xi['s'] *0 # returns correct shape
            for j, ens in enumerate(self.ens_mapping):
                if ens[:3] == 'a06':
                    output[j] = p['c0a06']
                elif ens[:3] == 'a09':
                    output[j] = p['c0a09']
                elif ens[:3] == 'a12':
                    output[j] = p['c0a12']
                elif ens[:3] == 'a15':
                    output[j] = p['c0a15']
                else:
                    output[j] = 0

        return output


    def fitfcn_nlo_ct(self, p, xi):
        output = (
            + p['k_l'] *xi['l'] 
            + p['k_s'] *xi['s'] 
            + p['k_a'] *xi['a']
        )
        return output


    def fitfcn_n2lo_ct(self, p, xi):     
        output = ( 
            + p['k_aa'] *xi['a'] *xi['a']
            + p['k_al'] *xi['a'] *xi['l']
            + p['k_as'] *xi['a'] *xi['s']
            + p['k_ll'] *xi['l'] *xi['l']
            + p['k_ls'] *xi['l'] *xi['s']
            + p['k_ss'] *xi['s'] *xi['s']
        )
        return output


    def fitfcn_n2lo_log(self, p, xi):
        output = p['k_ll_g'] *xi['l']**2 *np.log(xi['l'])
        return output


    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]





        
