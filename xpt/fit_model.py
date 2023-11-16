import lsqfit
import gvar as gv 
import copy
import numpy as np
import h5py as h5
import functools
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

# local modules
import xpt.non_analytic_functions as naf
import xpt.xi_fit as xi
import xpt.i_o as i_o

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
                 strange:str,
                 **kwargs
                ):
        self.data = data
        self.prior = prior
        if self.data is None:
            raise ValueError('you need to pass data to the fitter')
        self._phys_pt_data = phys_pt_data
        self.strange = strange
        self._model_info = self.fetch_models()
        self.options = kwargs
        # default values for optional params 
        self.svd_test = self.options.get('svd_test', False)
        self.svd_tol = self.options.get('svd_tol', None)
        self.mdl_key = self.options.get('mdl_key',None)
        self.model_info = self._model_info[self.mdl_key]

        # for key,value in kwargs.items():
        #     if key == 'svd_test':
        #         self.svd_test = value
        #     if key == 'svd_tol':
        #         self.svd_tol = int(value)
        #     if key == 'extrapolate':
        #         self.extrap = bool(value)

        self._posterior = None
        hbarc =  197.3269804, # MeV-fm
        self.models, self.models_dict = self._make_models()

        lam_sigma_particles = ['lambda', 'sigma', 'sigma_st']
        xi_particles = ['xi','xi_st']
        y_particles = ['xi','xi_st','lambda', 'sigma', 'sigma_st']

        if 'lambda' in self.model_info['particles']:
            self.data_subset = {part : self.data['m_'+part] for part in lam_sigma_particles}
        if 'xi' in self.model_info['particles'] :
            self.data_subset = {part:self.data['m_'+part] for part in xi_particles}
            if self.model_info['units'] == 'fpi':
                # self.y = {part: self.data['m_'+part]*data['lam_chi']*data['a_fm'] for part in xi_particles}
                self.y = {part: self.data['m_'+part] for part in xi_particles}

            if self.model_info['units'] == 'phys':
                # self.y = {part: self.data['m_'+part]*data['a_fm']/hbarc for part in xi_particles}
                self.y = {part: self.data['m_'+part] for part in xi_particles}

        if 'proton' in self.model_info['particles'] :
            self.data_subset = {'proton':self.data['m_proton']}
            if self.model_info['units'] == 'fpi':
                self.y = {'proton': data['m_proton']*data['lam_chi']*data['a_fm']}
            if self.model_info['units'] == 'phys':
                self.y = {'proton':data['m_proton']*data['a_fm']/hbarc}

    def update_svd_tol(self,new_svd_tol):
        self.svd_tol = new_svd_tol
    
    def fetch_models(self):
        with open('../xpt/models_test.yaml', 'r') as f:
            _models = yaml.load(f, Loader=yaml.FullLoader)
        models = {}
        if self.strange == '2':
            models = _models['models']['xi']
        elif self.strange == '1':
            models = _models['models']['lam']    
        elif self.strange == '0':
            models = _models['models']['proton']
        return models


    def format_extrapolation(self):
        """formats the extrapolation dictionary"""
        extrapolation_data = self.extrapolation()
        pdg_mass = {
            'xi': gv.gvar(1314.86,20),
            'xi_st': gv.gvar(1531.80,32),
            'lambda': gv.gvar(1115.683,6),
            'sigma': gv.gvar(1192.642,24),
            'sigma_st': gv.gvar(1383.7,1.0)
        }
        output = ""
        for particle, data in extrapolation_data.items():
            output += f"Particle: {particle}\n"
            measured = pdg_mass[particle]
            output += f"{data} [PDG: {measured}]\n"
            output += "---\n"

        return output
    
    def extrapolation(self, p=None, data=None, xdata=None):
        '''chiral extrapolations of baryon mass data using the Feynman-Hellmann theorem to quantify pion mass and strange quark mass dependence of baryon masses. Extrapolations are to the physical point using PDG data. 
        Returns:
            - extrapolated mass (meV)
        '''

        if p is None:
            p = self.get_posterior
        if data is None:
            data = self._phys_pt_data
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        if xdata is None:
            xdata = {}
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['units'] == 'fpi':
            xdata['eps_pi'] = p['eps_pi']
        p['l3_bar'] = -1/4 * (
            gv.gvar('3.53(26)') + np.log(xdata['eps_pi']**2))
        p['l4_bar'] =  gv.gvar('4.73(10)')
        p['c2_F'] = gv.gvar(0,20)
        p['c1_F'] = gv.gvar(0,20)
         
        MULTIFIT_DICT = {
            'xi': xi.Xi,
            'xi_st': xi.Xi_st,
        }
        results = {}

        for particle in self.model_info['particles']:
            model_class = MULTIFIT_DICT.get(particle)
            if model_class is not None:
                model_instance = model_class(datatag=particle, model_info=self.model_info)
        
                results[particle] = {}
                output = 0
                # extrapolate hyperon mass to the physical point 
                if self.model_info['units'] == 'lattice':
                    for particle in self.model_info['particles']:
                        output+= model_instance.fitfcn(p=p) * self.phys_point_data['hbarc']
                if self.model_info['units'] == 'fpi':
                    output+= model_instance.fitfcn(p=p) * self.phys_point_data['lam_chi']
                if self.model_info['units'] == 'phys':
                    output+= model_instance.fitfcn(p=p) 
            results[particle] = output
        return results

        
    def __str__(self):
        return f"{str(self.format_extrapolation())},{self.fit}"


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
    
    @property
    def get_posterior(self):
        return self._get_posterior()
    
    def _get_posterior(self,param=None):
        if param is not None:
            return self.fit.p[param]
        elif param == 'all':
            return self.fit.p
        output = {}
        for key in self.prior:
            if key in self.fit.p:
                output[key] = self.fit.p[key]
        return output

    
    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        model_array = np.array([])
        model_dict = {}

        if 'xi' in model_info['particles']:
            xi_model = xi.Xi(datatag='xi', model_info=model_info)
            model_array = np.append(model_array, xi_model)
            model_dict['xi'] = xi_model

        if 'xi_st' in model_info['particles']:
            xi_st_model = xi.Xi_st(datatag='xi_st', model_info=model_info)
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
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('light', self.model_info['order_light']), ('disc', self.model_info['order_disc']),
                             ('strange', self.model_info['order_strange']), ('xpt', self.model_info['order_chiral']),
                             ('fpi',self.model_info['order_fpi'])]:
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
        
        # this is "converting" the pseudoscalars into priors so that they do not count as data #
        if self.model_info['order_strange'] is not None:
            new_prior['m_k'] = data['m_k']
        if self.model_info['order_light'] is not None:
            new_prior['eps2_a'] = data['eps2_a']
            new_prior['m_pi'] = data['m_pi']
            new_prior['lam_chi'] = data['lam_chi']
            new_prior['eps_pi'] = data['eps_pi']
            # new_prior['a_fm'] = data['a_fm']
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
            for lec_type in ['disc', 'light', 'strange', 'xpt','fpi']:
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
            output['xi']['lo']['xpt'] = ['B_{xi,2}']

            output['xi']['nlo']['xpt'] = [
                'g_{xi,xi}', 'g_{xi_st,xi}', 'm_{xi_st,0}']
            output['xi']['n2lo']['disc'] = ['d_{xi,aa}', 'd_{xi,al}', ]
            output['xi']['n2lo']['strange'] = [
                'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            output['xi']['n2lo']['light'] = ['b_{xi,4}','B_{xi,4}']
            output['xi']['n2lo']['xpt'] = ['a_{xi,4}', 's_{xi,bar}']
            output['xi']['n2lo']['fpi'] = ['A_{xi,4}','c0']

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
