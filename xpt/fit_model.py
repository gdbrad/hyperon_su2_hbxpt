import lsqfit
import gvar as gv 
import copy
import numpy as np
import h5py as h5
import functools
from pathlib import Path
import matplotlib.pyplot as plt

# local modules
import xpt.non_analytic_functions as naf
import xi_fit


class FitModel:
    """
    The `FitModel` class is designed to fit models to data using least squares fitting.
    It takes in the model information, and options for empirical Bayes analysis.

    Attributes:
       
        model_info (dict):  information about the model to be fit.
        empbayes (bool): A boolean indicating whether to perform empirical Bayes analysis.
        empbayes_grouping (list): A list of dictionaries containing information about how to group the data
                                  for the empirical Bayes analysis.
        _fit (tuple): A tuple containing the fit results. The first element is a dictionary of the fit
                      parameters, and the second element is a dictionary of the fit errors.
        _posterior (dict): information about the posterior distribution of the fit.
        _empbayes_fit (tuple):empirical Bayes fit results. The first element is the
                               empirical Bayes prior, and the second element is a dictionary of the fit
                               parameters.

    Methods:
        __init__(self, prior, data, model_info, empbayes, empbayes_grouping):
            Constructor method for the `FitRoutine` class.
        __str__(self):
            String representation of the fit.
        fit(self):
            Method to perform the fit. Returns the fit results as a tuple.
        _make_models(self):
            Method to create models.
        _make_prior(self):
            Method to create prior information.
    """

    def __init__(self,
                 data:dict,
                 prior:dict,
                 phys_pt_data:dict,
                 fit_options:list,
                 model_info:dict,
                ):

        self.prior = prior
        self.data = data
        if self.data is None:
            raise ValueError('you need to pass data to the fitter')
        self._phys_pt_data = phys_pt_data
        
        self.model_info = model_info.copy()
        self.svd_test = self.fit_options['svd_test']
        self.svd_tol = self.model_info['svd_tol']
        self.scheme = self.model_info['eps2a_defn']
        self.units = self.model_info['units']
        self._posterior = None
        hbarc =  197.3269804, # MeV-fm
        self.models, self.models_dict = self._make_models()
    
        
    def __str__(self):
        return str(self.fit)


    @functools.cached_property
    def fit(self):
        prior_final = self._make_prior()
        data = self.y
        fitter = lsqfit.MultiFitter(models=self.models)
        if self.svd_test:
            svd_cut = self.input_output.perform_svdcut()
            fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False,svdcut=svd_cut)
            # fig = plt.figure('svd_diagnosis', figsize=(7, 4))
            # for ens in self.ensembles:
            #     svd_test.plot_ratio(show=True)
        else:
            # if self.svd_tol is None:
            #     fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False)
            # else:
            
            fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False,svdcut=self.svd_tol)

        return fit
    
    @property
    def fit_info(self):
        fit_info = {}
        fit_info = {
            'name' : self.model_info['name'],
            'logGBF' : self.fit.logGBF,
            'chi2/df' : self.fit.chi2 / self.fit.dof,
            'Q' : self.fit.Q,
            'phys_point' : self.phys_point_data,
            # 'error_budget' : self.error_budget,
            'prior' : self.prior,
            'posterior' : self.posterior
        }
        return fit_info
    
    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        model_array = np.array([])
        model_dict = {}

        if 'xi' in model_info['particles']:
            xi_model = xi_fit.Xi(datatag='xi', model_info=model_info)
            model_array = np.append(model_array, xi_model)
            model_dict['xi'] = xi_model

        if 'xi_st' in model_info['particles']:
            xi_st_model = xi_fit.Xi_st(datatag='xi_st', model_info=model_info)
            model_array = np.append(model_array, xi_st_model)
            model_dict['xi_st'] = xi_st_model

        return model_array, model_dict 
    
    def _make_prior(self, data=None,z=None,scale_data=None):
        '''
        Only need priors for LECs/data needed in fit.
        Separates all parameters that appear in the hyperon extrapolation formulae 
        '''
        if data is None:
            data = self.data
        # print(list(data))
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('light', self.model_info['order_light']), ('disc', self.model_info['order_disc']),
                             ('strange', self.model_info['order_strange']), ('xpt', self.model_info['order_chiral'])]:
                # include all orders equal to and less than desired order in expansion #
                if value == 'llo':
                    orders = ['llo']
                elif value == 'lo':
                    orders = ['llo', 'lo']
                elif value == 'nlo':
                    orders = ['llo', 'lo', 'nlo']
                elif value == 'n2lo':
                    orders = ['llo', 'lo', 'nlo', 'n2lo']
                elif value == 'n4lo':
                    orders = ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']
                else:
                    orders = []

                for o in orders:
                    keys.extend(self._get_prior_keys(
                        particle=p, order=o, lec_type=l))
        for key in keys:
            new_prior[key] = prior[key]
        
        if self.model_info['order_strange'] is not None:
            new_prior['m_k'] = data['m_k']
        if self.model_info['order_light'] is not None:
            new_prior['eps2_a'] = data['eps2_a']
            new_prior['m_pi'] = data['m_pi']
            new_prior['lam_chi'] = data['lam_chi']
            new_prior['eps_pi'] = data['eps_pi']
            new_prior['a_fm'] = data['a_fm']
        if self.model_info['fv']:
            new_prior['L'] = data['L']

        if self.model_info['order_disc'] is not None:
            new_prior['lam_chi'] = data['lam_chi']

        if z is None:
            return new_prior
        zkeys = self._empbayes_grouping()

        for k in new_prior:
            for group in zkeys:
                if k in zkeys[group]:
                    new_prior[k] = gv.gvar(0, np.exp(z[group]))
        return new_prior

    def _get_prior_keys(self, particle='all', order='all', lec_type='all'):
        if particle == 'all':
            output = []
            for particle in ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st','proton','delta']:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif lec_type == 'all':
            output = []
            for lec_type in ['disc', 'light', 'strange', 'xpt']:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        else:
        # construct dict of lec names corresponding to particle, order, lec_type #
            output = {}
            for p in ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st','proton','delta']:
                output[p] = {}
                for o in ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']:
                    output[p][o] = {}

            
            output['proton']['llo' ]['light'  ] = ['m_{proton,0}']
            output['proton']['lo'  ]['disc'   ] = ['d_{proton,a}']
            output['proton']['lo'  ]['light'  ] = ['b_{proton,2}','l4_bar','c0']
            output['proton']['lo'  ]['strange'] = ['d_{proton,s}']
            output['proton']['lo'  ]['xpt']     = ['l4_bar']
            output['proton']['nlo' ]['xpt'    ] = ['g_{proton,proton}', 'g_{proton,delta}','m_{delta,0}']
            output['proton']['n2lo']['disc'   ] = ['d_{proton,aa}', 'd_{proton,al}']
            output['proton']['n2lo']['strange'] = ['d_{proton,as}', 'd_{proton,ls}','d_{proton,ss}']
            output['proton']['n2lo']['light'  ] = ['b_{proton,4}']
            output['proton']['n2lo']['xpt'    ] = ['a_{proton,4}', 'g_{proton,4}']
            output['proton']['n4lo']['disc'   ] = ['d_{proton,all}', 'd_{proton,aal}']
            output['proton']['n4lo']['strange'] = []
            output['proton']['n4lo']['light'  ] = ['b_{proton,6}']
            output['proton']['n4lo']['xpt'    ] = []

            output['lambda']['llo']['light'] = ['m_{lambda,0}']
            output['lambda']['lo']['disc'] = ['d_{lambda,a}']
            output['lambda']['lo']['strange'] = ['d_{lambda,s}']
            output['lambda']['lo']['light'] = ['s_{lambda}','S_{lambda}']
            output['lambda']['nlo']['xpt'] = [
                'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'm_{sigma,0}', 'm_{sigma_st,0}']
            output['lambda']['n2lo']['disc'] = [
                'd_{lambda,aa}', 'd_{lambda,al}']
            output['lambda']['n2lo']['strange'] = [
                'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}']
            output['lambda']['n2lo']['light'] = ['b_{lambda,4}','B_{lambda,4}']
            output['lambda']['n2lo']['xpt'] = ['a_{lambda,4}', 's_{sigma}', 's_{sigma,bar}']

            output['sigma']['llo']['light'] = ['m_{sigma,0}']
            output['sigma']['lo']['disc'] = ['d_{sigma,a}']
            output['sigma']['lo']['strange'] = ['d_{sigma,s}']
            output['sigma']['lo']['light'] = ['s_{sigma}','S_{sigma}']
            output['sigma']['nlo']['xpt'] = [
                'g_{sigma,sigma}', 'g_{lambda,sigma}', 'g_{sigma_st,sigma}', 'm_{lambda,0}', 'm_{sigma_st,0}']
            output['sigma']['n2lo']['disc'] = [
                'd_{sigma,aa}', 'd_{sigma,al}', ]
            output['sigma']['n2lo']['strange'] = [
                'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}']
            output['sigma']['n2lo']['light'] = ['b_{sigma,4}','B_{sigma,4}']
            output['sigma']['n2lo']['xpt'] = [
                'a_{sigma,4}', 's_{lambda}', 's_{sigma,bar}']

            output['sigma_st']['llo']['light'] = ['m_{sigma_st,0}']
            output['sigma_st']['lo']['disc'] = ['d_{sigma_st,a}']
            output['sigma_st']['lo']['strange'] = ['d_{sigma_st,s}']
            output['sigma_st']['lo']['light'] = ['s_{sigma,bar}','S_{sigma,bar}']
            output['sigma_st']['nlo']['xpt'] = [
                'g_{sigma_st,sigma_st}', 'g_{lambda,sigma_st}', 'g_{sigma_st,sigma}', 'm_{lambda,0}', 'm_{sigma,0}']
            output['sigma_st']['n2lo']['disc'] = [
                'd_{sigma_st,aa}', 'd_{sigma_st,al}']
            output['sigma_st']['n2lo']['strange'] = [
                'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}']
            output['sigma_st']['n2lo']['light'] = ['b_{sigma_st,4}']
            output['sigma_st']['n2lo']['xpt'] = ['a_{sigma_st,4}', 's_{sigma}','S_{sigma}']

            output['xi']['llo']['light'] = ['m_{xi,0}']
            output['xi']['lo']['disc'] = ['d_{xi,a}']
            output['xi']['lo']['light'] = ['s_{xi}','S_{xi}']
            output['xi']['lo']['strange'] = ['d_{xi,s}']
            output['xi']['lo']['xpt'] = ['c0','B_{xi,2}']

            output['xi']['nlo']['xpt'] = [
                'g_{xi,xi}', 'g_{xi_st,xi}', 'm_{xi_st,0}']
            output['xi']['n2lo']['disc'] = ['d_{xi,aa}', 'd_{xi,al}', ]
            output['xi']['n2lo']['strange'] = [
                'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            output['xi']['n2lo']['light'] = ['b_{xi,4}','B_{xi,4}']
            output['xi']['n2lo']['xpt'] = ['a_{xi,4}', 's_{xi,bar}']

            output['xi_st']['llo']['light'] = ['m_{xi_st,0}']
            output['xi_st']['lo']['disc'] = ['d_{xi_st,a}']
            output['xi_st']['lo']['light'] = ['s_{xi,bar}','S_{xi,bar}']
            output['xi_st']['lo']['strange'] = ['d_{xi_st,s}']
            output['xi_st']['nlo']['xpt'] = [
                'g_{xi_st,xi_st}', 'g_{xi_st,xi}', 'm_{xi,0}']
            output['xi_st']['n2lo']['disc'] = ['d_{xi_st,aa}', 'd_{xi_st,al}']
            output['xi_st']['n2lo']['strange'] = [
                'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
            output['xi_st']['n2lo']['light'] = ['b_{xi_st,4}','B_{xi_st,4}']
            output['xi_st']['n2lo']['xpt'] = ['a_{xi_st,4}', 's_{xi}']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []
