import copy
import lsqfit
import numpy as np
import h5py as h5
import gvar as gv
import pprint
import sys
import os
import functools
# local modules
import xpt.non_analytic_functions as naf
import xpt.i_o as i_o


class FitRoutine:
    """
    The `FitRoutine` class is designed to fit models to data using least squares fitting.
    It takes in the prior information, data, model information, and options for empirical Bayes analysis.

    Attributes:
        prior (dict): prior information for the fit.
        data (dict):  data to be fit. The keys correspond to the names of the
                     data types(baryon correlators), and the values are arrays of the data.
        model_info (dict):  information about the model to be fit.
        empbayes (bool): A boolean indicating whether to perform empirical Bayes analysis.
        empbayes_grouping (list): A list of dictionaries containing information about how to group the data
                                  for the empirical Bayes analysis.
        _fit (tuple): A tuple containing the fit results. The first element is a dictionary of the fit
                      parameters, and the second element is a dictionary of the fit errors.
        _simultaneous (bool): A boolean indicating whether to fit data simultaneously.
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

    def __init__(self, prior, data, model_info,phys_point_data,emp_bayes,empbayes_grouping,fv=None):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
        # TODO REPLACE WITH NEW BS FILE 
        with h5.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])
        # TODO REPLACE WITH UPDATED SCALE SETTING FILE 
        with h5.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        ensembles.remove('a12m220')
        ensembles.remove('a12m220ms')
        ensembles.remove('a12m310XL')
        ensembles.remove('a12m220S')
        ensembles.remove('a12m180L')
        self.ensembles = ensembles
        self.prior = prior
        self.data = data
        self.model_info = model_info.copy()
        self._simultaneous = False
        self._posterior = None
        self._phys_point_data = phys_point_data
        self.emp_bayes = False #boolean
        self.empbayes_grouping = None #groups for empirical bayes prior study
        self._empbayes_fit = None
        self.fv = None # TODO not yet implemented 
        # this is manually reconstructing the gvar to decorrelate x and y data
        y_particles = ['lambda', 'sigma', 'sigma_st', 'xi_st', 'xi']
        data_subset = {part : self.data['m_'+part] for part in y_particles}
        self.y = gv.gvar(dict(gv.mean(data_subset)),dict(gv.evalcov(data_subset)))
        # x_data_subset = {ens
        # key: value for key, value in self.data.items() if key not in data_subset
        # }
        # self.x = x_data_subset
    
    def __str__(self):
        return str(self.fit)

    @functools.cached_property
    def fit(self):
        models = self._make_models()
        prior = self._make_prior()
        data = self.y
        fitter = lsqfit.MultiFitter(models=models)
        fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)
    
        return fit

    def _empbayes_grouping(self):
        '''
        routine adapted from @millernb
        '''
        zkeys = {}
        if self.empbayes_grouping == 'all':
            for param in self.prior:
                zkeys[param] = [param]
        # include particle choice xi or xi_st to fill inside bracket
        elif self.empbayes_grouping == 'order':
            # vary all light quark terms together, strange terms together
            zkeys['chiral_llo'] = ['m_{xi,0}', 'm_{xi_st,0}']
            zkeys['chiral_lo'] = ['s_{xi}', 's_{xi,bar}']
            zkeys['chiral_nlo'] = ['g_{xi,xi}',
                                   'g_{xi_st,xi}', 'g_{xi_st,xi_st}']
            zkeys['chiral_n2lo'] = [
                'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
            zkeys['disc_nlo'] = ['d_{xi,a}',
                                 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}']
            zkeys['disc_n2lo'] = [
                'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']

        # discretization effects
        elif self.empbayes_grouping == 'disc':
            zkeys['chiral'] = ['m_{xi,0}', 'm_{xi_st,0}', 's_{xi}', 's_{xi,bar}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}',
                               'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
            zkeys['disc'] = ['d_{xi,a}', 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}',
                             'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            
        elif self.empbayes_grouping == 'disc_only':
            zkeys['disc'] = ['d_{xi,a}', 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}',
                             'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
                             'd_{xi_st,aa}','d_{xi_st,al}' ,'d_{xi_st,as}','d_{xi_st,ls}', 'd_{xi_st,ss}']
            
        all_keys = [k for g in zkeys for k in zkeys[g]]
        prior_keys = list(self._make_prior())
        ignored_keys = set(all_keys) - set(prior_keys)

        # Don't determine empirical priors in param not in model
        for group in zkeys:
            for key in ignored_keys:
                if key in ignored_keys and key in zkeys[group]:
                    zkeys[group].remove(key)

        return zkeys

    def _make_empbayes_fit(self, empbayes_grouping='disc_only',observable=None):
        if (self._empbayes_fit is None) or (empbayes_grouping != self.empbayes_grouping):
            self.empbayes_grouping = empbayes_grouping
            self._counter = {'iters' : 0, 'evals' : 0}

            z0 = gv.BufferDict()
            for group in self._empbayes_grouping():
                z0[group] = 1.0

            def analyzer(arg):
                self._counter['evals'] += 1
                print("\nEvals: ", self._counter['evals'], arg,"\n")
                print(type(arg[0]))
                return None
            models = self._make_models()
            fitter = lsqfit.MultiFitter(models=models)

            fit, z = fitter.empbayes_fit(z0, fitargs=self._make_fitargs, maxit=20, analyzer=analyzer,tol=0.1)
            self._empbayes_fit = fit

        return self._empbayes_fit

    def _make_fitargs(self, z):
        '''
        preparing fit args that will be passed to fitter.empbayes_fit
        '''
        data = self.data
        prior = self._make_prior(z=z)

        # Ideally:
        # Don't bother with more than the hundredth place
        # Don't let z=0 (=> null GBF)
        # Don't bother with negative values (meaningless)
        # But for some reason, these restrictions (other than the last) cause empbayes_fit not to converge
        # multiplicity = {}
        # for key in z:
        #     multiplicity[key] = 0
        #     z[key] = np.abs(z[key])

        # Helps with convergence (minimizer doesn't use extra digits -- bug in lsqfit?)
        def sig_fig(x): return np.around(
            x, int(np.floor(-np.log10(x))+3))  # Round to 3 sig figs

        def capped(x, x_min, x_max): return np.max([np.min([x, x_max]), x_min])

        zkeys = self._empbayes_grouping()
        zmin = 1e-2
        zmax = 1e3
        for group in z.keys():
            for param in prior.keys():
                if param in zkeys[group]:
                    z[group] = sig_fig(capped(z[group], zmin, zmax))
                    prior[param] = gv.gvar(0, 1) * z[group]

        return dict(data=data,prior=prior)

    @property
    def posterior(self):
        return self._get_posterior()

    def _get_posterior(self,param=None):
        if param == 'all':
            return self.fit.p
        if param is not None:
            return self.fit.p[param]
        
        output = {}
        for param in self.prior:
            if param in self.fit.p:
                output[param] = self.fit.p[param]
        return output  
        
    @property
    def phys_point_data(self):
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self._phys_point_data
        else:
            return self._phys_point_data[parameter]
        
    def shift_latt_to_phys(self,ens=None,phys_params=None,observable=None,debug=None):
        '''shift fitted values of the observable(hyperon) on each lattice to a 
        new sector of parameter space in which all parameters are fixed except
        the physical parameter of interest,eg. eps2_a (lattice spacing), eps_pi (pion mass), etc. '''
        value_shifted = {}
        vals = {}
        for j, ens_j in enumerate(self.ensembles):
            if ens is None or ens_j == ens:
                # vals['latt'] = {}

                # for part in model_info['particles']:
                vals['latt'] = self.fit.y[observable][j]
                vals['fit'] = self.extrapolate_to_ens(ens=ens_j)
                vals['phys'] = self.extrapolate_to_ens(ens=ens_j, phys_params=phys_params)
                value_fit_phys = self.extrapolate_to_ens(ens=ens_j, phys_params=phys_params)
                value_latt = self.fit.y[observable][j]
                value_fit = self.extrapolate_to_ens(ens=ens_j)
                if debug:
                    print(vals['latt'],'latt')
                    print(vals['fit'] ,'fit')
                    print(vals['phys'],'phys')
                
                value_shifted[ens_j] = value_latt +  (value_fit_phys[observable] - value_fit[observable])
                if ens is not None:
                    return value_shifted[ens_j]
        return value_shifted
        
    

    def extrapolate_to_ens(self,ens=None, phys_params=None):
        if phys_params is None:
            phys_params = []
        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xdata = {}
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fit.p:
                    shape = self.fit.p[param].shape
                    if param in phys_params:
                        posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc']
                    elif shape == ():
                        posterior[param] = self.fit.p[param]
                    else:
                        posterior[param] = self.fit.p[param][j]
                if 'eps_pi' in phys_params:
                    xdata['eps_pi'] = self.phys_point_data['m_pi'] / self.phys_point_data['lam_chi']
                if 'd_eps2_s' in phys_params:
                    xdata['d_eps2_s'] = (2 *self.phys_point_data['m_k']**2 - self.phys_point_data['m_pi']**2)/ self.phys_point_data['lam_chi']**2
                if 'eps2_a' in phys_params:
                    xdata['eps_a'] = 0
                if ens is not None:
                    return self.get_fitfcn(p=posterior, data={},xdata=xdata)
                extrapolated_values[j] = self.get_fitfcn(p=posterior, data={}, xdata=xdata)
        return extrapolated_values
            

    def get_fitfcn(self,p=None,data=None,particle=None,xdata=None):
        output = {}
        if p is None:
            # p = copy.deepcopy(self.posterior)
            p = {}
            p.update(self.posterior)
        if data is None:
            data = copy.deepcopy(self.phys_point_data)
        p.update(data)
        # print(p,'p')
        for mdl in self._make_models(model_info=self.model_info):
            part = mdl.datatag
            output[part] = mdl.fitfcn(p=p,data=data,xdata=xdata)
        if particle is None:
            return output
        
        return output[particle]

    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'xi' in model_info['particles']:
            models = np.append(models, Xi(datatag='xi', model_info=model_info))

        if 'xi_st' in model_info['particles']:
            models = np.append(models, Xi_st(
                datatag='xi_st', model_info=model_info))

        if 'lambda' in model_info['particles']:
            models = np.append(models, Lambda(
                datatag='lambda', model_info=model_info))

        if 'sigma' in model_info['particles']:
            models = np.append(models, Sigma(
                datatag='sigma', model_info=model_info))

        if 'sigma_st' in model_info['particles']:
            models = np.append(models, Sigma_st(
                datatag='sigma_st', model_info=model_info))

        return models

    def _make_prior(self, data=None,z=None,scale_data=None):
        '''
        Only need priors for LECs/data needed in fit.
        verbosely separates all parameters that appear in the hyperon extrapolation formulae 

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
            for particle in ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st']:
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
            for p in ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st']:
                output[p] = {}
                for o in ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']:
                    output[p][o] = {}

            output['lambda']['llo']['light'] = ['m_{lambda,0}']
            output['lambda']['lo']['disc'] = ['d_{lambda,a}']
            output['lambda']['lo']['strange'] = ['d_{lambda,s}']
            output['lambda']['lo']['light'] = ['s_{lambda}']
            output['lambda']['nlo']['xpt'] = [
                'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'm_{sigma,0}', 'm_{sigma_st,0}']
            output['lambda']['n2lo']['disc'] = [
                'd_{lambda,aa}', 'd_{lambda,al}']
            output['lambda']['n2lo']['strange'] = [
                'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}']
            output['lambda']['n2lo']['light'] = ['b_{lambda,4}']
            output['lambda']['n2lo']['xpt'] = ['a_{lambda,4}', 's_{sigma}', 's_{sigma,bar}']

            output['sigma']['llo']['light'] = ['m_{sigma,0}']
            output['sigma']['lo']['disc'] = ['d_{sigma,a}']
            output['sigma']['lo']['strange'] = ['d_{sigma,s}']
            output['sigma']['lo']['light'] = ['s_{sigma}']
            output['sigma']['nlo']['xpt'] = [
                'g_{sigma,sigma}', 'g_{lambda,sigma}', 'g_{sigma_st,sigma}', 'm_{lambda,0}', 'm_{sigma_st,0}']
            output['sigma']['n2lo']['disc'] = [
                'd_{sigma,aa}', 'd_{sigma,al}', ]
            output['sigma']['n2lo']['strange'] = [
                'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}']
            output['sigma']['n2lo']['light'] = ['b_{sigma,4}']
            output['sigma']['n2lo']['xpt'] = [
                'a_{sigma,4}', 's_{lambda}', 's_{sigma,bar}']

            output['sigma_st']['llo']['light'] = ['m_{sigma_st,0}']
            output['sigma_st']['lo']['disc'] = ['d_{sigma_st,a}']
            output['sigma_st']['lo']['strange'] = ['d_{sigma_st,s}']
            output['sigma_st']['lo']['light'] = ['s_{sigma,bar}']
            output['sigma_st']['nlo']['xpt'] = [
                'g_{sigma_st,sigma_st}', 'g_{lambda,sigma_st}', 'g_{sigma_st,sigma}', 'm_{lambda,0}', 'm_{sigma,0}']
            output['sigma_st']['n2lo']['disc'] = [
                'd_{sigma_st,aa}', 'd_{sigma_st,al}']
            output['sigma_st']['n2lo']['strange'] = [
                'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}']
            output['sigma_st']['n2lo']['light'] = ['b_{sigma_st,4}']
            output['sigma_st']['n2lo']['xpt'] = ['a_{sigma_st,4}', 's_{sigma}']

            output['xi']['llo']['light'] = ['m_{xi,0}']
            output['xi']['lo']['disc'] = ['d_{xi,a}']
            output['xi']['lo']['light'] = ['s_{xi}']
            output['xi']['lo']['strange'] = ['d_{xi,s}']
            output['xi']['nlo']['xpt'] = [
                'g_{xi,xi}', 'g_{xi_st,xi}', 'm_{xi_st,0}']
            output['xi']['n2lo']['disc'] = ['d_{xi,aa}', 'd_{xi,al}', ]
            output['xi']['n2lo']['strange'] = [
                'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            output['xi']['n2lo']['light'] = ['b_{xi,4}']
            output['xi']['n2lo']['xpt'] = ['a_{xi,4}', 's_{xi,bar}']

            output['xi_st']['llo']['light'] = ['m_{xi_st,0}']
            output['xi_st']['lo']['disc'] = ['d_{xi_st,a}']
            output['xi_st']['lo']['light'] = ['s_{xi,bar}']
            output['xi_st']['lo']['strange'] = ['d_{xi_st,s}']
            output['xi_st']['nlo']['xpt'] = [
                'g_{xi_st,xi_st}', 'g_{xi_st,xi}', 'm_{xi,0}']
            output['xi_st']['n2lo']['disc'] = ['d_{xi_st,aa}', 'd_{xi_st,al}']
            output['xi_st']['n2lo']['strange'] = [
                'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
            output['xi_st']['n2lo']['light'] = ['b_{xi_st,4}']
            output['xi_st']['n2lo']['xpt'] = ['a_{xi_st,4}', 's_{xi}']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []
            
    # def interpolate_w0a(self, latt_spacing, simultaneous_interpolation=False):
    #     return self.fitfcn_interpolation(latt_spacing=latt_spacing, simultaneous_interpolation=simultaneous_interpolation, observable='w0')


class Xi(lsqfit.MultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Xi baryon
    '''
    def __init__(self, datatag, model_info):
        super(Xi, self).__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata = None):
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        if xdata is None:
            xdata = {}
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        # xdata['eps_a'] = ((1/2) * p['a/w'])
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513
    
        # llo
        output = p['m_{xi,0}']
        output += self.fitfcn_lo_ct(p, xdata) 
        output += self.fitfcn_nlo_xpt(p, xdata) 
        output += self.fitfcn_n2lo_ct(p, xdata) 
        output += self.fitfcn_n2lo_xpt(p, xdata) 

        return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        output = 0
        if self.model_info['fit_phys_units']: # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{xi,0}'] * (p['d_{xi,a}'] * xdata['eps2_a'])
                # print(output)

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi}'] * xdata['lam_chi'] * xdata['eps_pi']**2)
                # print(output,"2")

            if self.model_info['order_strange'] is not None:
                if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo'] and 'd_{xi,s}' in p:
                    output += p['m_{xi,0}']*(p['d_{xi,s}'] * xdata['d_eps2_s'])

        elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output +=  (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output += (p['d_{xi,s}'] * xdata['d_eps2_s'])

        return output

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        if self.model_info['xpt']:
            if self.model_info['fit_phys_units']:  # lam_chi dependence ON #
                output = (
                    (xdata['lam_chi'] * (-3/2) * np.pi * p['g_{xi,xi}']**2 * xdata['eps_pi']**3)
                    - (p['g_{xi_st,xi}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta']))
                )
            elif self.model_info['fit_fpi_units']:  # lam_chi dependence OFF #
                output = (
                    ((-3/2) * np.pi * p['g_{xi,xi}'] ** 2 * xdata['eps_pi'] ** 3)
                    - (p['g_{xi_st,xi}'] ** 2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta']))
                )

        if self.model_info['xpt'] is False:
            return 0
        return output


    def fitfcn_n2lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['fit_phys_units']:  # lam_chi dependence ON #
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{xi,0}'] * (
                    (p['d_{xi,as}'] * xdata['eps2_a']) *
                    (xdata['d_eps2_s']) +
                    (p['d_{xi,ls}'] * (xdata['d_eps2_s']) *
                    (xdata['eps_pi'] ** 2) +
                    (p['d_{xi,ss}'] * (xdata['d_eps2_s'] ** 2)
                    )))

            if self.model_info['order_disc'] in ['n2lo']:
                output +=p['m_{xi,0}']*(
                    (p['d_{xi,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2)+(p['d_{xi,aa}'] * xdata['eps2_a'])**2)

            if self.model_info['order_light'] in ['n2lo']:
                output += (xdata['eps_pi']**4 * p['b_{xi,4}'] * xdata['lam_chi']) 

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * xdata['eps_pi']**4 *np.log(xdata['eps_pi']**2) * p['a_{xi,4}']

        elif self.model_info['fit_fpi_units']:  # lam_chi dependence ON #
            if self.model_info['order_strange'] in ['n2lo']:
                output += (
                    (p['d_{xi,as}'] * xdata['eps2_a']) *
                    (xdata['d_eps2_s']) +
                    (p['d_{xi,ls}'] * (xdata['d_eps2_s']) *
                    (xdata['eps_pi'] ** 2) +
                    (p['d_{xi,ss}'] * (xdata['d_eps2_s'] ** 2)
                    )))

            if self.model_info['order_disc'] in ['n2lo']:
                output += (
                    + (p['d_{xi,al}'] * (xdata['eps2_a']) * xdata['eps_pi'] ** 2)
                    + p['d_{xi,aa}'] * xdata['eps2_a'] ** 2)

            if self.model_info['order_chiral'] in ['n2lo']:
                output += (
                (xdata['eps_pi'] ** 4) * p['b_{xi,4}']) + (xdata['lam_chi'] * (xdata['eps_pi'] ** 4)) *np.log(xdata['eps_pi'] ** 2) * p['a_{xi,4}']

        return output
    
    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        if self.model_info['xpt']:
            if self.model_info['fit_phys_units']: # lam_chi dependence ON #
                output = (
                (3/2) * p['g_{xi_st,xi}'] ** 2 * (p['s_{xi}'] - p['s_{xi,bar}']) *
                xdata['lam_chi'] * xdata['eps_pi'] ** 2 *
                naf.fcn_J(xdata['eps_pi'], xdata['eps_delta']) 
        )
            elif self.model_info['fit_fpi_units']:  # lam_chi dependence OFF #
                output = (
                    (3/2) * p['g_{xi_st,xi}'] ** 2 * (p['s_{xi}'] - p['s_{xi,bar}']) *
                    xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta']) +
                    ((xdata['eps_pi'] ** 4)) * np.log(xdata['eps_pi'] ** 2) * p['a_{xi,4}']
                )

        if self.model_info['xpt'] is False:
            return 0
        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]


class Xi_st(lsqfit.MultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Xi baryon
    '''
    def __init__(self, datatag, model_info):
        super(Xi_st, self).__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        '''extraplation formulae'''
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        if xdata is None:
            xdata = {}
        if 'lc' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        # xdata['eps_a'] = ((1/2) * p['a/w'])
        if 'a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (
                (2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513

        output = p['m_{xi_st,0}'] #llo
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^2) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['fit_phys_units']: # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += (p['m_{xi_st,0}'] * (p['d_{xi_st,a}']*xdata['eps2_a']))

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi,bar}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None:
                if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo'] and 'd_{xi_st,s}' in p:
                    output += p['m_{xi_st,0}']*(p['d_{xi_st,s}'] * xdata['d_eps2_s'])
                    
        elif self.model_info['fit_fpi_units']: # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += (p['d_{xi_st,a}']*xdata['eps2_a'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi,bar}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output += (p['d_{xi_st,s}'] * xdata['d_eps2_s'])

        return output

    # no nlo disc or strange terms

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        output = 0
        if self.model_info['xpt']:
            if self.model_info['fit_phys_units']: # lam_chi dependence ON #
                output = ((xdata['lam_chi'] *
                        (-5/6) * np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi']**3)
                        - ((1/2)*p['g_{xi_st,xi}']**2 * xdata['lam_chi']*naf.fcn_F(xdata['eps_pi'], -xdata['eps_delta'])))
            elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
                 output += (-5/6) * np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi']**3
                 output -= 1/2*p['g_{xi_st,xi}']**2 * naf.fcn_F(xdata['eps_pi'], -xdata['eps_delta'])

        elif self.model_info['xpt'] is False:
            return 0
        return output

    # n2lo terms

    def fitfcn_n2lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['fit_phys_units']: # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['n2lo']:
                output += (p['m_{xi_st,0}']*(
                    (p['d_{xi_st,aa}'] * xdata['eps2_a']**2) +
                    (p['d_{xi_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2)
                ))

            if self.model_info['order_strange'] in ['n2lo']:
                output += (
                    p['m_{xi_st,0}']*(
                        (p['d_{xi_st,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']) +
                        (p['d_{xi_st,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) +
                        (p['d_{xi_st,ss}'] * xdata['d_eps2_s']**2)))

            if self.model_info['order_light'] in ['n2lo']:
                output += (xdata['eps_pi']**4 * p['b_{xi_st,4}'] * xdata['lam_chi']) 

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * xdata['eps_pi']**4 *np.log(xdata['eps_pi']**2) * p['a_{xi_st,4}']

        elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
            if self.model_info['order_disc'] in ['n2lo']:
                output += (p['m_{xi_st,0}']*(
                    (p['d_{xi_st,aa}'] * xdata['eps2_a']**2) +
                    (p['d_{xi_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2)
                )
                )
            if self.model_info['order_strange'] in ['n2lo']:
                output += (
                    p['m_{xi_st,0}']*(
                        (p['d_{xi_st,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']) +
                        (p['d_{xi_st,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) +
                        (p['d_{xi_st,ss}'] * xdata['d_eps2_s']**2)))

            if self.model_info['order_chiral'] in ['n2lo']:
                output += (
                    xdata['eps_pi']**4 * p['b_{xi_st,4}'])
        
        return output

    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        output = 0
        if self.model_info['xpt']:
            if self.model_info['fit_phys_units']: # lam_chi dependence ON #
                output += (
                    ((3/4) * p['g_{xi_st,xi}']**2 * (p['s_{xi,bar}']-p['s_{xi}']) *
                    xdata['lam_chi'] * xdata['eps_pi']**2 *
                    naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])) +
                    (xdata['lam_chi'] * (xdata['eps_pi']**4)) *
                    np.log(xdata['eps_pi']**2) * p['a_{xi_st,4}']
            )
            elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
                output += (
                    ((3/4) * p['g_{xi_st,xi}']**2 * (p['s_{xi,bar}']-p['s_{xi}']) *xdata['eps_pi']**2 *
                    naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])) +
                    (xdata['eps_pi']**4)) *np.log(xdata['eps_pi']**2) * p['a_{xi_st,4}']

        elif self.model_info['xpt'] is False:
            return 0

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

# Strangeness=1 Hyperons

class Lambda(lsqfit.MultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Lambda baryon
    '''
    def __init__(self, datatag, model_info):
        super(Lambda, self).__init__(datatag)
        self.model_info = model_info

    # fit_data from i_o module
    def fitfcn(self, p, data=None):
        '''extraplation formulae'''
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        xdata = {}
        # xdata['m_k'] = p['m_k']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        xdata['eps_sigma_st'] = (
            p['m_{sigma_st,0}'] - p['m_{lambda,0}']) / p['lam_chi']
        xdata['eps_sigma'] = (
            p['m_{sigma,0}'] - p['m_{lambda,0}']) / p['lam_chi']
        xdata['eps2_a'] = p['eps2_a']
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (
                (2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513
        xdata['lam_chi'] = p['lam_chi']

        # not-even leading order
        output =  p['m_{lambda,0}']
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        output = 0
        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
            output += (p['m_{lambda,0}'] *
                       (p['d_{lambda,a}'] * xdata['eps2_a']))

        if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
            output += (p['m_{lambda,0}'] *
                       (p['d_{lambda,s}'] * xdata['d_eps2_s']))

        if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
            output += (p['s_{lambda}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

        return output

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        if self.model_info['xpt']:
            output = (xdata['lam_chi'] * (-1/2) * p['g_{lambda,sigma}']**2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma'])
                      - (2 * p['g_{lambda,sigma_st}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st'])))
            return output
        if self.model_info['xpt'] is False:
            return 0

    def fitfcn_n2lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^4)'''
        output = 0
        if self.model_info['order_disc'] in ['n2lo']:
            output += (p['m_{lambda,0}']*(
                (p['d_{lambda,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2)
                + (p['d_{lambda,aa}'] * xdata['eps2_a']**2)))

        if self.model_info['order_strange'] in ['n2lo']:
            output += p['m_{lambda,0}']*(
                (p['d_{lambda,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']) +
                (p['d_{lambda,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) +
                (p['d_{lambda,ss}'] * xdata['d_eps2_s']**2)
            )

        if self.model_info['order_light'] in ['n2lo']:
            output += (
                xdata['eps_pi']**4 * p['b_{lambda,4}']*xdata['lam_chi'])
            
        if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * xdata['eps_pi']**4 *np.log(xdata['eps_pi']**2) * p['a_{lambda,4}']

        return output

    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        if self.model_info['xpt']:
            term_1_coeff = (3/4) * p['g_{lambda,sigma}']**2 * (p['s_{lambda}'] - p['s_{sigma}'])
            term_1 = term_1_coeff * xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma'])

            term_2_coeff = 3 * p['g_{lambda,sigma_st}']**2 * (p['s_{lambda}'] - p['s_{sigma,bar}'])
            term_2 = term_2_coeff * xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])

            # term_3 = xdata['lam_chi'] * xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{lambda,4}']

            output = term_1 + term_2
            return output

        if self.model_info['xpt'] is False:
            return 0

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Sigma(lsqfit.MultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Sigma baryon
    '''
    def __init__(self, datatag, model_info):
        super(Sigma, self).__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None):
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        xdata = {}
        # xdata['m_k'] = p['m_k']
        xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        xdata['eps_lambda'] = (
            p['m_{sigma,0}'] - p['m_{lambda,0}']) / p['lam_chi']
        xdata['eps_sigma_st'] = (
            p['m_{sigma_st,0}'] - p['m_{sigma,0}']) / p['lam_chi']
        # xdata['eps_a'] = ((1/2) * p['a/w'])
        xdata['eps2_a'] = p['eps2_a']
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (
                (2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513

        output = p['m_{sigma,0}'] #llo
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^2) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
            output += (p['m_{sigma,0}'] * (p['d_{sigma,a}'] * xdata['eps2_a']))

        if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
            output += (p['s_{sigma}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

        if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
            output += (p['m_{sigma,0}']*(p['d_{sigma,s}'] * xdata['d_eps2_s']))

        return output

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        if self.model_info['xpt'] is True:
            output = (
                (xdata['lam_chi'] * (-np.pi) *
                 p['g_{sigma,sigma}']**2 * xdata['eps_pi']**3)
                - ((1/6) * p['g_{lambda,sigma}']**2 * xdata['lam_chi']
                   * naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda']))
                - ((2/3) * p['g_{sigma_st,sigma}']**2 * xdata['lam_chi']
                   * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st']))
            )
        elif self.model_info['xpt'] is False:
            return 0
        return output

    def fitfcn_n2lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['order_strange'] in ['n2lo']:
            output += p['m_{sigma,0}']*(
                p['d_{sigma,as}'] * (xdata['eps2_a']) *
                (xdata['d_eps2_s']) +
                (p['d_{sigma,ls}'] * xdata['d_eps2_s'] *
                 xdata['eps_pi']**2) +
                (p['d_{sigma,ss}'] * xdata['d_eps2_s'] ** 2)
            )
        if self.model_info['order_disc'] in ['n2lo']:
            output += p['m_{sigma,0}']*(
                (p['d_{sigma,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2+ 
                (p['d_{sigma,aa}'] * xdata['eps2_a']**2)
                ))

        if self.model_info['order_light'] in ['n2lo']:
            output += (
                xdata['eps_pi']**4 * p['b_{sigma,4}']*xdata['lam_chi'])
            
        if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * xdata['eps_pi']**4 *np.log(xdata['eps_pi']**2) * p['a_{sigma,4}']

        return output

    # extract lecs in quotes and insert into prior dict in hyperon_fit??
    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        if self.model_info['xpt']:
            output = (
                p['g_{sigma_st,sigma}']**2*(p['s_{sigma}']-p['s_{sigma,bar}']) * xdata['lam_chi'] * xdata['eps_pi']**2 *
                naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
                + (1/4)*p['g_{lambda,sigma}']**2 * (p['s_{sigma}'] -
                                                    p['s_{lambda}']) * xdata['lam_chi'] * xdata['eps_pi']**2
                * naf.fcn_J(xdata['eps_pi'], -xdata['eps_lambda'])
            )
            return output
        if self.model_info['xpt'] is False:
            return 0

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Sigma_st(lsqfit.MultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the sigma* baryon
    '''
    def __init__(self, datatag, model_info):
        super(Sigma_st, self).__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None):
        '''extrapolation formulae'''
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        xdata = {}
        # xdata['m_k'] = p['m_k']
        xdata['lam_chi'] = p['lam_chi']
        xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        xdata['eps_lambda'] = (
            p['m_{sigma_st,0}'] - p['m_{lambda,0}']) / p['lam_chi']
        xdata['eps_sigma'] = (
            p['m_{sigma_st,0}'] - p['m_{sigma,0}']) / p['lam_chi']
        # xdata['eps_a'] = ((1/2) * p['a/w'])
        xdata['eps2_a'] = p['eps2_a']
        xdata['d_eps2_s'] = (2 * p['m_k']**2 - p['m_pi'] **
                             2) / p['lam_chi']**2 - 0.3513
       # not-even leading order
        output = p['m_{sigma_st,0}']
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        output = 0
        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
            output += (p['m_{sigma_st,0}'] *
                       (p['d_{sigma_st,a}'] * xdata['eps2_a']))

        if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
            output += (p['s_{sigma,bar}'] * xdata['lam_chi']
                       * xdata['eps_pi']**2)

        if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
            output += (p['m_{sigma_st,0}'] *
                       p['d_{sigma_st,s}'] * xdata['d_eps2_s'])

        return output

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        if self.model_info['xpt'] is True:
            output = (
                (xdata['lam_chi'] * ((-5/9)*np.pi) *
                 p['g_{sigma_st,sigma_st}']**2 * xdata['eps_pi']**3)
                - (1/3) * p['g_{sigma_st,sigma}']**2 * xdata['lam_chi'] *
                naf.fcn_F(xdata['eps_pi'], -xdata['eps_sigma'])
                - (1/3) * p['g_{lambda,sigma_st}']**2 * xdata['lam_chi'] *
                naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda'])
            )
        if self.model_info['xpt'] is False:
            return 0
        return output

    def fitfcn_n2lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^4)'''
        output = 0

        if self.model_info['order_strange'] in ['n2lo']:
            output += p['m_{sigma_st,0}']*(
                # term 2
                (p['d_{sigma_st,as}'] * xdata['eps2_a']) *

                (xdata['d_eps2_s']) +
                # term 3
                (p['d_{sigma_st,ls}'] * xdata['d_eps2_s'] *
                 xdata['eps_pi']**2) +
                # term 4
                (p['d_{sigma_st,ss}'] * xdata['d_eps2_s']**2)
            )

        if self.model_info['order_disc'] in ['n2lo']:
            output += p['m_{sigma_st,0}']*(
                (p['d_{sigma_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2)
                + (p['d_{sigma_st,aa}'] * xdata['eps2_a']**2))

        if self.model_info['order_light'] in ['n2lo']:
            output += (
                xdata['lam_chi'] * xdata['eps_pi']**4 * p['b_{sigma_st,4}'])
            
        if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * xdata['eps_pi']**4 *np.log(xdata['eps_pi']**2) * p['a_{sigma_st,4}']

        return output

    # extract lecs in quotes and insert into prior dict in hyperon_fit??
    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        if self.model_info['xpt']:
            output = (
                (1/2)*p['g_{sigma_st,sigma}']**2 * (p['s_{sigma,bar}']-p['s_{sigma}']) *
                xdata['lam_chi'] * xdata['eps_pi']**2 *
                (naf.fcn_J(xdata['eps_pi'], -xdata['eps_sigma']))
                + (1/2)*p['g_{lambda,sigma_st}']**2 * (p['s_{sigma,bar}'] -
                                                       p['s_{sigma}']) * xdata['lam_chi'] * xdata['eps_pi']**2
                * naf.fcn_J(xdata['eps_pi'], -xdata['eps_lambda'])
            )
            return output
        if self.model_info['xpt'] is False:
            return 0

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]
