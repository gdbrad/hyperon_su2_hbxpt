import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
import non_analytic_functions as naf
import i_o

class fit_routine(object):

    def __init__(self, prior, data, model_info):
         
        self.prior = prior
        self.data = data
        self.model_info = model_info.copy()
        self._fit = None
        self._simultaneous = False

        self.empbayes = None
        self._empbayes_fit = None
        self.y = {datatag : self.data['m_'+datatag] for datatag in self.model_info['particles']}
        #need to save self.y to generate fit , correlated with self.y
        

    def __str__(self):
        return str(self.fit)

    @property
    def fit(self):
        if self._fit is None:
            models = self._make_models()
            prior = self._make_prior()
            data = self.y

            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)

            self._fit = fit

        return self._fit

    def _empbayes(self):
        zkeys = {}

        if self.empbayes == 'all':
            for param in self.prior:
                zkeys[param] = [param]
        
        # include particle choice xi or xi_st to fill inside bracket
        elif self.empbayes == 'order':
            zkeys['chiral_n0lo'] = ['m_{xi,0}', 'm_{xi_st,0}']
            zkeys['chiral_lo']   = ['s_{xi}'  , 's_{xi,bar}']
            zkeys['chiral_nlo']  = ['g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}']
            zkeys['chiral_n2lo'] = ['b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
            zkeys['latt_nlo']    = ['d_{xi,a}', 'd_{xi_st,a}','d_{xi,s}', 'd_{xi_st,s}'] 
            zkeys['latt_n2lo']   = ['d_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}','d_{xi,ss}'] 
        
        # discretization effects
        # could just zip the chiral and latt dicts above...
        elif self.empbayes == 'disc':
            zkeys['chiral'] = ['m_{xi,0}', 'm_{xi_st,0}','s_{xi}' , 's_{xi,bar}','g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}',
                               'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
            zkeys['disc']   = ['d_{xi,a}', 'd_{xi_st,a}','d_{xi,s}', 'd_{xi_st,s}',
                               'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}','d_{xi,ss}']

        all_keys = np.array([k for g in zkeys for k in zkeys[g]])
        prior_keys = list(self._make_prior())

        return zkeys

    def _make_empbayes_fit(self, empbayes_grouping='order'):
        if (self._empbayes_fit is None) or (empbayes != self.empbayes):
            self.empbayes = empbayes

            z0 = gv.BufferDict()
            for group in self._empbayes():
                z0[group] = 1.0

            # Might need to change minargs default values for empbayes_fit to converge:
            # tol=1e-8, svdcut=1e-12, debug=False, maxit=1000, add_svdnoise=False, add_priornoise=False
            # Note: maxit != maxfev. See https://github.com/scipy/scipy/issues/3334
            # For Nelder-Mead algorithm, maxfev < maxit < 3 maxfev?

            # For debugging. Same as 'callback':
            # https://github.com/scipy/scipy/blob/c0dc7fccc53d8a8569cde5d55673fca284bca191/scipy/optimize/optimize.py#L651

            fit, z = lsqfit.empbayes_fit(z0, fitargs=self._make_fitargs, maxit=200, analyzer=None)
            print(z)
            self._empbayes_fit = fit

        return self._empbayes_fit

    def _make_fitargs(self, z):
        data = self.data
        prior = self._make_prior()

        # Ideally:
            # Don't bother with more than the hundredth place
            # Don't let z=0 (=> null GBF)
            # Don't bother with negative values (meaningless)
        # But for some reason, these restrictions (other than the last) cause empbayes_fit not to converge
        multiplicity = {}
        for key in z:
            multiplicity[key] = 0
            z[key] = np.abs(z[key])


        # Helps with convergence (minimizer doesn't use extra digits -- bug in lsqfit?)
        sig_fig = lambda x : np.around(x, int(np.floor(-np.log10(x))+3)) # Round to 3 sig figs
        capped = lambda x, x_min, x_max : np.max([np.min([x, x_max]), x_min])

        zkeys = self._empbayes()
        zmin = 1e-2
        zmax = 1e3
        for group in z.keys():
            for param in prior.keys():
                if param in zkeys[group]:
                    z[group] = sig_fig(capped(z[group], zmin, zmax))
                    prior[param] = gv.gvar(0, 1) *z[group]


        
        fitfcn = self._make_models()[-1].fitfcn
        #print(self._counter['iters'], ' ', z)#{key : np.round(1. / z[key], 8) for key in z.keys()}
        
        return (dict(data=data, fcn=fitfcn, prior=prior))


    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'xi' in model_info['particles']:
            models = np.append(models,Xi(datatag='xi', model_info=model_info))

        if 'xi_st' in model_info['particles']:
            models = np.append(models,Xi_st(datatag='xi_st', model_info=model_info))

        if 'delta' in model_info['particles']:
            models = np.append(models,Delta(datatag='delta', model_info=model_info))

        if 'lam' in model_info['particles']:
            models = np.append(models,Lambda(datatag='lam', model_info=model_info))

        if 'sigma' in model_info['particles']:
            models = np.append(models,Sigma(datatag='sigma', model_info=model_info))

        if 'sigma_st' in model_info['particles']:
            models = np.append(models,Sigma_st(datatag='sigma_st', model_info=model_info))
        
        return models


    def _make_prior(self, data=None):
        if data is None:
            data = self.data
        prior = self.prior
        new_prior = {}
        for key in prior:
            new_prior[key] = prior[key]
        for key in ['m_pi', 'm_k', 'lam_chi', 'a/w']:
            new_prior[key] = data[key]
        return new_prior

    
class Xi(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Xi, self).__init__(datatag)

        # override build data and build prior methods in lsqfit 
        # two models, Xi and Xi*, models need to know part of data to each model use datatag
        self.model_info = model_info

    #fit_data from i_o module
    def fitfcn(self, p, data=None):
        
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        eps_pi = p['m_pi'] / p['lam_chi']
        eps_delta = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        eps_a = (1/2) * p['a/w']
        
        #not-even leading order
        output = p['m_{xi,0}']

        #lo

        if self.model_info['order_disc'] in ['lo','nlo','n2lo']:
            output += self.fitfcn_lo_ct(p,eps_a)

        if self.model_info['order_chiral'] in ['lo', 'nlo','n2lo']:
            output  += self.fitfcn_lo_xpt(p,eps_pi) 
            
        if self.model_info['order_strange'] in ['lo', 'nlo','n2lo']:
            output  += self.fitfcn_lo_strange(p)

        #nlo
        if self.model_info['order_chiral'] in ['nlo', 'n2lo']:
            output += self.fitfcn_nlo_xpt(p, eps_pi, eps_delta)

        #n2lo

        if self.model_info['order_disc'] in ['n2lo']:
            output += self.fitfcn_n2lo_ct(p, eps_a, eps_pi)

        if self.model_info['order_chiral'] in ['n2lo']:
            output += self.fitfcn_n2lo_xpt(p, eps_pi, eps_delta)

        if self.model_info['order_strange'] in ['n2lo']:
            # self.model_info['pp'] == True:
            # return 
            output += self.fitfcn_n2lo_strange(p)

        #just log terms and special terms 
        if self.model_info['xpt'] is True:
            output += (
                  self.fitfcn_lo_xpt(p,eps_pi)
                + self.fitfcn_nlo_xpt(p, eps_pi, eps_delta)
                + self.fitfcn_n2lo_xpt(p, eps_pi, eps_delta)
            )
        return output


    def fitfcn_lo_ct(self, p, eps_a):
        
        output = p['m_{xi,0}'] * (p['d_{xi,a}'] * (eps_a**2))

        return output

    def fitfcn_lo_xpt(self, p, eps_pi):
        if self.model_info['xpt'] is True:
            output = p['s_{xi}'] * p['lam_chi'] * eps_pi**2
            return output

        if self.model_info['xpt'] is False:
            return 0

    def fitfcn_lo_strange(self,p):
    #first term of red expansion 
        output = ( 
            p['d_{xi,s}'] *  
            ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)
        )
        return output

    # no nlo disc terms 

    def fitfcn_nlo_xpt(self,p, eps_pi, eps_delta):
        if self.model_info['xpt'] is True:
            output = p['lam_chi'] *(
            -(3/2) *np.pi *p['g_{xi,xi}']**2 *eps_pi**3
            - p['g_{xi_st,xi}']**2 *naf.fcn_F(eps_pi, eps_delta))
            return output
        if self.model_info['xpt'] is False:
            return 0


    # n2lo terms
    def fitfcn_n2lo_ct(self, p, eps_a, eps_pi):     
        output = p['m_{xi,0}'] * ( 
            + (p['d_{xi,al}'] * eps_a**2 * eps_pi**2)
            + p['d_{xi,aa}'] * eps_a**4
            + (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**4)) *p['b_{xi,4}'] ##excluding ln(eps_pi^2)
        )
        
        return output
    
    def fitfcn_n2lo_log(self,p):
        output = (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**3))
        
        return output
        

    def fitfcn_n2lo_xpt(self,p, eps_pi, eps_delta):
        if self.model_info['xpt'] is True:
            output = (
                (3/2) * (p['g_{xi_st,xi}']**2)*(p['s_{xi}']-p['s_{xi,bar}']) *
                (p['lam_chi'] * eps_pi**2) * 
                (naf.fcn_J(eps_pi,eps_delta)) +
                (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**4)) *np.log(eps_pi**2)*p['a_{xi,4}']
            )
            return output
        if self.model_info['xpt'] is False:
            return 0


    def fitfcn_n2lo_strange(self,p):
        output = (
            #term 2
            p['d_{xi,as}']* (0.5 * p['a/w']**2) *

            ((2*p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) +
            #term 3
            ( p['d_{xi,ls}'] * ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)) * 
            ( p['m_pi'] / p['lam_chi']**2 ) +
            #term 4
            (p['d_{xi,ss}'] * ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)**2)
        )

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]


class Xi_st(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Xi_st, self).__init__(datatag)
        # override build data and build prior methods in lsqfit 
        # two models, Xi and Xi*, models need to know part of data to each model use datatag
        self.model_info = model_info

    #fit_data from i_o module
    def fitfcn(self, p, data=None):
        
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        eps_pi = p['m_pi'] / p['lam_chi']
        eps_delta = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        eps_a = (1/2) * p['a/w']
        
        #not-even leading order
        output = p['m_{xi_st,0}']

        #lo

        if self.model_info['order_disc'] in ['lo','nlo','n2lo']:
            output += self.fitfcn_lo_ct(p,eps_a)

        if self.model_info['order_chiral'] in ['lo', 'nlo','n2lo']:
            output  += self.fitfcn_lo_xpt(p,eps_pi) 
            
        if self.model_info['order_strange'] in ['lo', 'nlo','n2lo']:
            output  += self.fitfcn_lo_strange(p)

        #nlo
        if self.model_info['order_chiral'] in ['nlo', 'n2lo']:
            output += self.fitfcn_nlo_xpt(p, eps_pi, eps_delta)

        #n2lo

        if self.model_info['order_disc'] in ['n2lo']:
            output += self.fitfcn_n2lo_ct(p, eps_a, eps_pi)

        if self.model_info['order_chiral'] in ['n2lo']:
            output += self.fitfcn_n2lo_xpt(p, eps_pi, eps_delta)
            if self.model_info['include_log'] is True:
                output += self.fitfcn_n2lo_log(p, eps_pi)

        if self.model_info['order_strange'] in ['n2lo']:
            # self.model_info['pp'] == True:
            # return 
            output += self.fitfcn_n2lo_strange(p)

        #just log terms and special terms 
        if self.model_info['xpt'] is True:
            output += (
                  self.fitfcn_nlo_xpt(p, eps_pi, eps_delta)
                + self.fitfcn_n2lo_log(p,eps_pi)
            )
        return output


    def fitfcn_lo_ct(self, p, eps_a):
        
        output = p['m_{xi_st,0}'] * (p['d_{xi_st,a}'] * (eps_a**2))

        return output

    def fitfcn_lo_xpt(self, p, eps_pi):
        output = p['s_{xi}'] * p['lam_chi'] * eps_pi**2

        return output

    def fitfcn_lo_strange(self,p):
    #first term of red expansion 
        output = ( 
            p['d_{xi_st,s}'] *  
            ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)
        )
        return output

    # no nlo disc terms 

    def fitfcn_nlo_xpt(self,p, eps_pi, eps_delta):
        output = p['lam_chi'] *(
        -(3/2) *np.pi *p['g_{xi_st,xi_st}']**2 *eps_pi**3
        - p['g_{xi_st,xi}']**2 *naf.fcn_F(eps_pi, eps_delta))
        return output


    # n2lo terms
    def fitfcn_n2lo_ct(self, p, eps_a, eps_pi):     
        output = p['m_{xi_st,0}'] * ( 
            + (p['d_{xi_st,al}'] * eps_a**2 * eps_pi**2)
            + p['d_{xi_st,aa}'] * eps_a**4
            + (p['b_{xi_st,4}'] *(p['m_pi']**4 / p['lam_chi']**3)) ##excluding ln(eps_pi^2)
        )
        
        return output
    
    def fitfcn_n2lo_log(self,p, eps_pi):
        output = p['lam_chi'] * eps_pi**4 * np.log(eps_pi**2)
        
        return output
        

    def fitfcn_n2lo_xpt(self,p, eps_pi, eps_delta):
        output = (
            (3/2) * (p['g_{xi_st,xi}']**2)*(p['s_{xi}']-p['s_{xi,bar}']) *
            (p['lam_chi'] * eps_pi**2) * 
            (naf.fcn_J(eps_pi,eps_delta)) +
            (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**4)) * np.log(eps_pi**2) * p['a_{xi_st,4}']
        )
        return output


    def fitfcn_n2lo_strange(self,p):
        output = (
            #term 2
            p['d_{xi_st,as}']* (0.5 * p['a/w']**2) *

            ((2*p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) +
            #term 3
            ( p['d_{xi_st,ls}'] * ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)) * 
            ( p['m_pi'] / p['lam_chi']**2 ) +
            #term 4
            (p['d_{xi_st,ss}'] * ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)**2)
        )

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]

class Lambda(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Lambda, self).__init__(datatag)

        # override build data and build prior methods in lsqfit 
        # two models, Xi and Xi*, models need to know part of data to each model use datatag
        self.model_info = model_info

    #fit_data from i_o module
    def fitfcn(self, p, data=None):
        
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        eps_pi = p['m_pi'] / p['lam_chi']
        eps_delta = (p['m_{lam,0}'] - p['m_{sigma_st,0}']) / p['lam_chi']
        eps_a = (1/2) * p['a/w']
        
        #not-even leading order
        output = p['m_{lam,0}']

        #lo

        if self.model_info['order_disc'] in ['lo','nlo','n2lo']:
            output += self.fitfcn_lo_ct(p,eps_a)

        if self.model_info['order_chiral'] in ['lo', 'nlo','n2lo']:
            output  += self.fitfcn_lo_xpt(p,eps_pi) 
            
        if self.model_info['order_strange'] in ['lo', 'nlo','n2lo']:
            output  += self.fitfcn_lo_strange(p)

        #nlo
        if self.model_info['order_chiral'] in ['nlo', 'n2lo']:
            output += self.fitfcn_nlo_xpt(p, eps_pi, eps_delta)

        #n2lo

        if self.model_info['order_disc'] in ['n2lo']:
            output += self.fitfcn_n2lo_ct(p, eps_a, eps_pi)

        if self.model_info['order_chiral'] in ['n2lo']:
            output += self.fitfcn_n2lo_xpt(p, eps_pi, eps_delta)

        if self.model_info['order_strange'] in ['n2lo']:
            # self.model_info['pp'] == True:
            # return 
            output += self.fitfcn_n2lo_strange(p)

        #just log terms and special terms 
        if self.model_info['xpt'] is True:
            output += (
                  self.fitfcn_lo_xpt(p,eps_pi)
                + self.fitfcn_nlo_xpt(p, eps_pi, eps_delta)
                + self.fitfcn_n2lo_xpt(p, eps_pi, eps_delta)
            )
        return output


    def fitfcn_lo_ct(self, p, eps_a):
        
        output = p['m_{lam,0}'] * (p['d_{lam,a}'] * (eps_a**2))

        return output

    def fitfcn_lo_xpt(self, p, eps_pi):
        if self.model_info['xpt'] is True:
            output = p['s_{lam}'] * p['lam_chi'] * eps_pi**2
            return output

        if self.model_info['xpt'] is False:
            return 0

    def fitfcn_lo_strange(self,p):
    #first term of red expansion 
        output = ( 
            p['d_{lam,s}'] *  
            ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)
        )
        return output

    # no nlo disc terms 

    def fitfcn_nlo_xpt(self,p, eps_pi, eps_delta):
        if self.model_info['xpt'] is True:
            output = p['lam_chi'] *(
            +(3/2) *np.pi *p['g_{lam,sigma}']**2 *eps_pi**3
            - p['g_{xi_st,xi}']**2 *naf.fcn_F(eps_pi, eps_delta))
            return output
        if self.model_info['xpt'] is False:
            return 0


    # n2lo terms
    def fitfcn_n2lo_ct(self, p, eps_a, eps_pi):     
        output = p['m_{xi,0}'] * ( 
            + (p['d_{xi,al}'] * eps_a**2 * eps_pi**2)
            + p['d_{xi,aa}'] * eps_a**4
            + (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**4)) *p['b_{xi,4}'] ##excluding ln(eps_pi^2)
        )
        
        return output
    
    def fitfcn_n2lo_log(self,p):
        output = (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**3))
        
        return output
        

    def fitfcn_n2lo_xpt(self,p, eps_pi, eps_delta):
        if self.model_info['xpt'] is True:
            output = (
                (3/2) * (p['g_{xi_st,xi}']**2)*(p['s_{xi}']-p['s_{xi,bar}']) *
                (p['lam_chi'] * eps_pi**2) * 
                (naf.fcn_J(eps_pi,eps_delta)) +
                (p['lam_chi'] *(p['m_pi']**4 / p['lam_chi']**4)) *np.log(eps_pi**2)*p['a_{xi,4}']
            )
            return output
        if self.model_info['xpt'] is False:
            return 0


    def fitfcn_n2lo_strange(self,p):
        output = (
            #term 2
            p['d_{xi,as}']* (0.5 * p['a/w']**2) *

            ((2*p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) +
            #term 3
            ( p['d_{xi,ls}'] * ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)) * 
            ( p['m_pi'] / p['lam_chi']**2 ) +
            #term 4
            (p['d_{xi,ss}'] * ((2*p['m_k']**2- p['m_pi']**2) / p['lam_chi']**2)**2)
        )

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]