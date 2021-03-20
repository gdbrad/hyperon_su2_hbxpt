import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
import non_analytic_functions as naf

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
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)

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
                'Xpt': False
            }

            datatag = model_info_interpolation['name']
            models = np.append(models, model_interpolation(datatag=datatag, model_info=model_info_interpolation, ens_mapping=self._ensemble_mapping, observable=self.observable))
            if not simultaneous:
                return models

        datatag = model_info['name']
        models = np.append(models, model(datatag=datatag, model_info=model_info))

        return models
