import lsqfit
import numpy as np
import gvar as gv
import sys
import os

import non_analytic_functions as naf


class Xi(lsqfit.MultiFitterModel):
    def __init__(self,p):
        self.p = p

    def fit_fcn(self,xi,p):
        

    def fitfcn_lo_ct(self, p):
        output = p['m_{xi,0}'] * (1 + p['d_a'] * (0.5* (p['a/w'])**2))

        return output

    def fitfcn_nlo_xpt(self, p):





    

    def fitfcn_n2lo_ct(self, p):     
        output = (
            prior['d_{xi,al}'] *  
            + p['A_aa'] *xi['a'] *xi['a']
            + p['A_al'] *xi['a'] *xi['l']
            + p['A_as'] *xi['a'] *xi['s']
            + p['A_ll'] *xi['l'] *xi['l']
            + p['A_ls'] *xi['l'] *xi['s']
            + p['A_ss'] *xi['s'] *xi['s']
        )

        if self.debug:
            self.debug_table['n2lo_ct'] = output

        return output

    def fitfcn_n2lo_xpt(self,p):
        



    

class Xi_st(lsqfit.MultiFitterModel):
    def __init__(self,p):
        self.p = p 



class fitter(object):

    def __init__(self, prior, fit_data, model_info, observable, ensemble_mapping=None, prior_interpolation=None):
        self.prior = prior
        self.prior_interpolation = prior_interpolation
        self.fit_data = fit_data
        self.model_info = model_info.copy()
        self.observable = observable
        
        # attributes of fitter object to fill later
        self.empbayes_grouping = None
        self._counter = {'iters' : 0, 'evals' : 0} # To force empbayes_fit to converge?
        self._empbayes_fit = None
        self._fit = None
        self._fit_interpolation = None
        self._simultaneous = False
        self._y = None
        self._ensemble_mapping = ensemble_mapping # Necessary for LO t0, w0 interpolations 


    def __str__(self):
        return str(self.fit)

        @property
    def fit(self):
        if self._fit is None:
            models = self._make_models()
            y_data = {self.model_info['name'] : self.y}
            prior = self._make_prior()

            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=y_data, prior=prior, fast=False, mopt=False)

            self._fit = fit

        return self._fit


    #@property
    def fit_interpolation(self, simultaneous=None):
        if simultaneous is None:
            simultaneous = self._simultaneous

        if self._fit_interpolation is None or simultaneous != self._simultaneous:
            self._simultaneous = simultaneous
            #make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))
            #y_data = make_gvar(1 / self.fit_data['a/w'])

            make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))
            if self.observable == 'w0':
                data = {self.model_info['name']+'_interpolation' : 1 / make_gvar(self.fit_data['a/w']) }
            elif self.observable == 't0':
                data = {self.model_info['name']+'_interpolation' : make_gvar(self.fit_data['t/a^2']) }

            if simultaneous:
                data[self.model_info['name']] = self.y


            models = self._make_models(interpolation=True, simultaneous=simultaneous)
            prior = self._make_prior(interpolation=True, simultaneous=simultaneous)

            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)
            self._fit_interpolation = fit

        return self._fit_interpolation

        def _make_models(self, model_info=None, interpolation=False, y_data=None, simultaneous=False):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])
        if interpolation:

            model_info_interpolation = {
                'name' : model_info['name'] + '_interpolation',
                'chiral_cutoff': 'Fpi',
                'order': 'n2lo',
                'latt_ct': 'n2lo',
                'include_log': False,
                'include_log2': False,
                'include_fv': False,
                'include_alphas': False,
                'exclude': []
            }

            datatag = model_info_interpolation['name']
            models = np.append(models, model_interpolation(datatag=datatag, model_info=model_info_interpolation, ens_mapping=self._ensemble_mapping, observable=self.observable))
            if not simultaneous:
                return models

        datatag = model_info['name']
        models = np.append(models, model(datatag=datatag, model_info=model_info))

        return models

class Xi(lsqfit.MultiFitterModel):

    def __init__(self, datatag, model_info, **kwargs):
        super(model, self).__init__(datatag)


    def fitfcn(self, p):
        


        # lo
        output = p['c0']

        if self.debug:
            self.debug_table['lo_ct'] = output

        # nlo
        if self.model_info['order'] in ['nlo', 'n2lo', 'n3lo']:
            output += self.fitfcn_nlo_ct(p, xi)
            if self.model_info['include_alphas']:
                output += self.fitfcn_nlo_latt_alphas(p, xi)
                
        elif self.model_info['latt_ct'] in ['nlo', 'n2lo', 'n3lo']: 
            output += self.fitfcn_nlo_latt_ct(p, xi)
            if self.model_info['include_alphas']:
                output += self.fitfcn_nlo_latt_alphas(p, xi)

        # n2lo 
        if self.model_info['order'] in ['n2lo', 'n3lo']:
            output += self.fitfcn_n2lo_ct(p, xi)
            if self.model_info['include_log']:
                output += self.fitfcn_n2lo_log(p, xi)

        elif self.model_info['latt_ct'] in ['n2lo', 'n3lo']:
            output += self.fitfcn_n2lo_latt_ct(p, xi)

        # n3lo
        if self.model_info['order'] in ['n3lo']:
            output += self.fitfcn_n3lo_ct(p, xi)
            if self.model_info['include_log']:
                output += self.fitfcn_n3lo_log(p, xi)
            if self.model_info['include_log2']:
                output += self.fitfcn_n3lo_log_sq(p, xi)
                
        elif self.model_info['latt_ct'] in ['n3lo']:
            output += self.fitfcn_n3lo_latt_ct(p, xi)



        for key in self.model_info['exclude']:
            del(p[key])

        if debug:
            #print(gv.tabulate(self.debug_table))
            temp_string = ''
            for key in self.debug_table:
                temp_string +='  % .15f:  %s\n' %(gv.mean(self.debug_table[key]), key)
            temp_string +='   -----\n'
            temp_string +='  % .15f:  %s\n' %(gv.mean(output), 'total')
            print(temp_string)

        return output


    def fitfcn_nlo_ct(self, p, xi):
        output = p['A_l'] *xi['l'] + p['A_s'] *xi['s'] + p['A_a'] *xi['a']

        if self.debug:
            self.debug_table['nlo_ct'] = output

        return output


    def fitfcn_n2lo_ct(self, p, xi):     
        output = ( 
            + p['A_aa'] *xi['a'] *xi['a']
            + p['A_al'] *xi['a'] *xi['l']
            + p['A_as'] *xi['a'] *xi['s']
            + p['A_ll'] *xi['l'] *xi['l']
            + p['A_ls'] *xi['l'] *xi['s']
            + p['A_ss'] *xi['s'] *xi['s']
        )

        if self.debug:
            self.debug_table['n2lo_ct'] = output

        return output


    def fitfcn_n2lo_log(self, p, xi):
        if self.model_info['include_fv']:
            output = p['A_ll_g'] *xi['l']**2 *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
        else:
            output = p['A_ll_g'] *xi['l']**2 *np.log(xi['l'])

        if self.debug:
            self.debug_table['n2lo_log'] = output

        return output


    def fitfcn_nlo_latt_ct(self, p, xi):
        output = p['A_a'] *xi['a']

        if self.debug:
            self.debug_table['nlo_latt'] = output

        return output


    def fitfcn_n2lo_latt_ct(self, p, xi):
        output = p['A_aa'] *xi['a']**2

        if self.debug:
            self.debug_table['n2lo_latt'] = output

        return output


    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]

class Xi_st(lsqfit.MultiFitterModel):

    def __init__(self, datatag, model_info, ens_mapping=None, observable=None, **kwargs):
        super(model_interpolation, self).__init__(datatag)

        # Model info
        self.model_info = model_info
        self.ens_mapping = ens_mapping
        self.observable = observable


    def fitfcn(self, p, fit_data=None, xi=None, latt_spacing=None, observable=None):
        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]

        for key in self.model_info['exclude']:
            p[key] = 0


        # Variables
        if xi is None:
            xi = {}
        if 'l' not in xi:
            xi['l'] = (p['m_pi'] / p['lam_chi'])**2
        if 's' not in xi:
            xi['s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2

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





        
