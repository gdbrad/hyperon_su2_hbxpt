import lsqfit
import numpy as np
import gvar as gv
import time
import matplotlib
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
import os
import h5py

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

#internal xpt modules
import fit_routine.fit_routine as fit
import i_o

class fit_analysis(object):
    
    def __init__(self, phys_point_data, data=None, model_info=None, prior=None):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))


        with h5py.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        with h5py.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        ensembles.remove('a12m220')
        ensembles.remove('a12m220S')
        #data,ensembles = i_o.InputOutput.get_data(scheme='w0_imp')

        self.ensembles = ensembles
        self.model_info = model_info
        self.data = data
        self.fitter = {} # fill fitter dict with models based on scheme, use 'wo_imp' for now 
        self.fitter = fit
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = {}

    
    @property
    def fit(self):
        temp_fit = self.fit
        self._fit = temp_fit

        return self._fit
    
    @property
    def fit_info(self):
        fit_info = {}
        fit_info = {
            'name' : self.model,
            #'w0_imp' : self.w0,
            'logGBF' : self.fit.logGBF,
            'chi2/df' : self.fit.chi2 / self.fit.dof,
            'Q' : self.fit.Q,
            'phys_point' : self.phys_point_data,
            #'error_budget' : self.error_budget['w0'],
            'prior' : self.prior,
            'posterior' : self.posterior,
        }
        return fit_info

    # Returns names of LECs in prior/posterior
    @property
    def fit_keys(self):
        output = {}
        
        keys1 = list(self._input_prior.keys())
        keys2 = list(self.fit.p.keys())
        output = np.intersect1d(keys1, keys2)
        return output

    @property
    def model(self):
        return self.model_info['name']

    @property
    def phys_point_data(self):
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self.phys_point_data.copy()
        else:
            return self.phys_point_data[parameter]

    @property
    def posterior(self):
        return self._get_posterior()

    # Returns dictionary with keys fit parameters, entries gvar results
    def _get_posterior(self, param=None):
        output = {}
        if param is None:
            output = {param : self.fit.p[param] for param in self.fit_keys}
        elif param == 'all':
            output = self.fit.p
        else:
            output = self.fit.p[param]

        return output

    @property
    def prior(self):
        return self._get_prior()

    def _get_prior(self, param=None):
        output = {}
        if param is None:
            output = {param : self.fit.prior[param] for param in self.fit_keys}
        elif param == 'all':
            output = self.fit.prior
        else:
            output = self.fit.prior[param]

        return output

    def _extrapolate_to_ens(self, ens=None, phys_params=None):
        if phys_params is None:
            phys_params = []

        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xi = {}
            if ens is None or (ens is not None and ens_j == ens):
                if 'alpha_s' in phys_params:
                    posterior['alpha_s'] = self.phys_point_data['alpha_s']

                if 'xi_l' in phys_params:
                    xi['l'] = self.phys_point_data['m_pi']**2 / self.phys_point_data['lam_chi']**2
                if 'xi_s' in phys_params:
                    xi['s'] = (2 *self.phys_point_data['m_k']**2 - self.phys_point_data['m_pi']**2)/ self.phys_point_data['lam_chi']**2
                if 'xi_a' in phys_params:
                    xi['a'] = 0

                if ens is not None:
                    return self.fitfcn(posterior=posterior, data={}, xi=xi)
                else:
                    extrapolated_values[ens_j] = self.fitfcn(posterior=posterior, data={}, xi=xi)
        return extrapolated_values

    def fitfcn(self, p, data=None, particle=None):
        output = {}
        for mdl in self.fitter._make_models:
            part = mdl.datatag
            output[part] = mdl.fitfcn(p,data)

        if particle is None:
            return output
        else:
            return output[particle]

