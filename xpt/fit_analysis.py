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
import yaml
import sys
import datetime
sys.setrecursionlimit(10000)

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
import xpt.fit_routine as fit
import xpt.i_o


class fit_analysis(object):
    
    def __init__(self, phys_point_data, data=None, model_info=None, prior=None,verbose=None):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
        # TODO REPLACE WITH NEW BS FILE 
        with h5py.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])
        # TODO REPLACE WITH UPDATED SCALE SETTING FILE 
        with h5py.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
       
        self.ensembles = ensembles
        self.model_info = model_info
        self.verbose = verbose # print all data points
        self.data = data
        self.fitter = {}
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = {}
        self.fitter = fit.FitRoutine(prior=prior,data=data, model_info=model_info,
                    phys_point_data=phys_point_data, emp_bayes=None,empbayes_grouping=None)
        self.fit = self.fitter.fit
      
    def __str__(self):
        # output = "Model: %s" %(self.model)
        output = '\n---\n'
        if self.verbose:
            fit_str = self.fit.format(maxline=True,pstyle='vv')
        else:
            fit_str = self.fit.format(pstyle='m')
        output+= fit_str
        output+= str(self.extrapolated_mass)
        # output += '\nError Budget:\n'
        # max_len = np.max([len(key) for key in self.error_budget])
        # for key in {k: v for k, v in sorted(self.error_budget.items(), key=lambda item: item[1], reverse=True)}:
        #     output += '  '
        #     output += key.ljust(max_len+1)
        #     for particle in ['m_'+ p for p in self.model_info['particles']]:

        #         output += '{: .1%}\n'.format((self.error_budget[key]/self.extrapolated_mass[particle].sdev)**2).rjust(7)

        return output
    @property
    def error_budget(self):
        '''
        useful for analyzing the impact of the a priori uncertainties encoded in the prior
        '''
        return self._get_error_budget()

    def _get_error_budget(self, verbose=True,**kwargs):
        '''
        hardcoded list of chiral expansion parameters associated with each hyperon,
        calculates a parameter's relative contribution to the total error inherent 
        in the mass expansion
        '''
        output = None
        strange_keys = [
        'd_{lambda,s}','d_{sigma,s}', 'd_{sigma_st,s}', 'd_{xi,s}', 'd_{xi_st,s}',
        'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}', 'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}',
        'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
        'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
        
        chiral_keys = [
        's_{lambda}', 's_{sigma}', 's_{sigma,bar}', 's_{xi}', 's_{xi,bar}', 
        'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'g_{sigma,sigma}', 'g_{sigma_st,sigma}', 
        'g_{sigma_st,sigma_st}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}', 'b_{lambda,4}', 
        'b_{sigma,4}', 'b_{sigma_st,4}', 'b_{xi,4}', 'b_{xi_st,4}', 'a_{lambda,4}', 'a_{sigma,4}', 
        'a_{sigma_st,4}', 'a_{xi,4}', 'a_{xi_st,4}'] 
        
        disc_keys = [
        'd_{lambda,a}', 'd_{sigma,a}',  
        'd_{sigma_st,a}', 'd_{xi,a}',  'd_{xi_st,a}', 'd_{lambda,aa}', 'd_{lambda,al}', 
        'd_{sigma,aa}', 'd_{sigma,al}',  'd_{sigma_st,aa}', 'd_{sigma_st,al}', 
        'd_{xi,aa}', 'd_{xi,al}',  'd_{xi_st,aa}', 'd_{xi_st,al}']

        light_keys = [
            'm_{lambda,0}', 'm_{sigma,0}', 'm_{sigma_st,0}', 'm_{xi,0}', 'm_{xi_st,0}'
        ]
        phys_keys = list(self.phys_point_data)
        stat_keys = ['lam_chi']# Since the input data is correlated, only need a single variable as a proxy for all

        if verbose:
            if output is None:
                output = ''

            inputs = {}
            inputs.update({str(param)+' [disc]': self._input_prior[param] for param in disc_keys if param in self._input_prior})
            inputs.update({str(param)+' [xpt]': self._input_prior[param] for param in chiral_keys if param in self._input_prior})
            inputs.update({str(param)+ '[strange]': self._input_prior[param] for param in strange_keys if param in self._input_prior})
            inputs.update({str(param)+' [pp]': self.phys_point_data[param] for param in list(phys_keys)})
            inputs.update({'x [stat]' : self._input_prior[param] for param in stat_keys if param in self._input_prior})# , 'y [stat]' : self.fitter.fit.y})

            if kwargs is None:
                kwargs = {}
            kwargs.setdefault('percent', False)
            kwargs.setdefault('ndecimal', 10)
            kwargs.setdefault('verify', True)

            print(gv.fmt_errorbudget(outputs=self.extrapolated_mass, inputs=inputs, verify=True))
        else:
            output = {}
            for particle in ['m_'+ p for p in self.model_info['particles']]:

                output['disc'] = self.extrapolated_mass[particle].partialsdev(
                            [self.prior[key] for key in disc_keys if key in self.prior])
                
                output['chiral'] = self.extrapolated_mass[particle].partialsdev(
                            [self.prior[key] for key in chiral_keys if key in self.prior])
                
                output['pp'] = self.extrapolated_mass[particle].partialsdev(
                            [self.phys_point_data[key] for key in phys_keys if key in phys_keys])
                
                output['stat'] = self.extrapolated_mass[particle].partialsdev(
                            [self._get_prior(stat_keys),self.fitter.fit.y])
        return output

    @property
    def fit_info(self):
        fit_info = {}
        fit_info = {
            'name' : self.model,
            'logGBF' : self.fit.logGBF,
            'chi2/df' : self.fit.chi2 / self.fit.dof,
            'Q' : self.fit.Q,
            'phys_point' : self.phys_point_data,
            'error_budget' : self.error_budget,
            'prior' : self.prior,
            'posterior' : self.posterior
        }
        return fit_info

    @property
    def extrapolated_mass(self):
        '''returns mass of a hyperon extrapolated to the physical point'''
        mdls = fit.FitRoutine(prior=self.prior,data=self.data, model_info=self.model_info,
                    phys_point_data=self.phys_point_data, emp_bayes=None,empbayes_grouping=None)
                              
        extrapolated_mass = mdls.get_fitfcn(p=self.posterior, data=self.phys_point_data)
        return extrapolated_mass

    @property
    def fit_keys(self):
        output = {}
        keys1 = list(self._input_prior.keys())
        keys2 = list(self.fitter.fit.p.keys())
        output = np.intersect1d(keys1, keys2)
        return output

    @property
    def model(self):
        return self.model_info['name']

    @property
    def phys_point_data(self):
        '''returns dict of physial constants from the PDG'''
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self._phys_point_data
        else:
            return self._phys_point_data[parameter]

    @property
    def posterior(self):
        '''Returns dictionary with keys fit parameters'''
        return self._get_posterior()

    def _get_posterior(self,param=None):
        if param == 'all':
            return self.fitter.fit.p
        elif param is not None:
            return self.fitter.fit.p[param]
        else:
            output = {}
            for param in self._input_prior:
                if param in self.fitter.fit.p:
                    output[param] = self.fitter.fit.p[param]
            return output

    @property
    def prior(self):
        return self._get_prior()

    def _get_prior(self, param=None):
        output = {}
        if param is None:
            output = {param : self.fitter.fit.prior[param] for param in self.fit_keys}
        elif param == 'all':
            output = self.fitter.fit.prior
        else:
            output = self.fitter.fit.prior[param]

        return output
    
    def fitfcn(self, p=None, data=None, particle=None):
        output = {}
        if p is None:
            p = {}
            p.update(self.posterior)
        if data is None:
            data = self.phys_point_data
            
        p.update(data)

        for mdl in self.fitter._make_models():
            part = mdl.datatag
            output[part] = mdl.fitfcn(p)

        if particle is None:
            return output
        else:
            return output[particle]
    
    # def plot_fit(self,xparam=None, yparam=None):
    #     if yparam is None:
    #         yparam = 'm_xi'
    #     x_fit = {}
    #     y_fit = {}
    #     c = {}
    #     colors = {
    #         '06' : '#6A5ACD',
    #         '09' : '#51a7f9',
    #         '12' : '#70bf41',
    #         '15' : '#ec5d57',
    #     }

    #     baryon_latex = {
    #         'sigma': '\Sigma',
    #         'sigma_st': '\Sigma^*',
    #         'xi': '\Xi',
    #         'xi_st': '\Xi^*',
    #         'lam': '\Lambda'
    #     }
    #     print(self.fit.p)

    #     for i in range(len(self.ensembles)):
    #         for j, param in enumerate([xparam, yparam]):
    #             if param in baryon_latex.keys():
    #                 value = self.fit.y[yparam][i]
    #                 latex_baryon = baryon_latex[param]
    #                 label = f'$m_{{{latex_baryon}}}(MeV)$'

    #             elif param == 'eps_pi':
    #                 value = self.posterior['eps_pi'][i]
    #                 label = '$\epsilon_\pi$'
    #                 #min,max linspace

    #             elif param == 'eps2_a':
    #                 value = self.fit.p['eps2_a'][i]
    #                 label = '$\epsilon_a^2$'
    #             if j == 0:
    #                 x_fit[i] = value
    #                 xlabel = label
    #             elif j == 1:
    #                 y_fit[i] = value
    #                 ylabel = label
    #     min_max = lambda arr : (np.nanmin(arr), np.nanmax(arr))
    #     min_val, max_val = min_max(self.data['eps2_a'])

    #     eps2_a = np.linspace(gv.mean(min_val), gv.mean(max_val))

    #     posterior = {}
    #     posterior.update(self.fit.p)
    #     eps2_a = posterior['eps2_a']
    #     print(eps2_a,'eps')

    #     # y_fit = self.fitfcn(particle=yparam)

    #     pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
    #     y_fit = self.fit.y[yparam]

    #     plt.fill_between(gv.mean(eps2_a), pm(y_fit, -1), pm(y_fit, +1))
    #     plt.show()
    #     print(y_fit,value)

    #     for i in range(len(self.ensembles)):
    #         plt.errorbar(gv.mean(x_fit[i]), gv.mean(y_fit[i]),
    #                     xerr=gv.sdev(x_fit[i]), yerr=gv.sdev(y_fit[i]),
    #                     marker='o', mec='w', zorder=3, linestyle='')

    #         plt.grid()
    #     plt.xlabel(xlabel, fontsize=24)
    #     plt.ylabel(ylabel, fontsize=24)
    #     plt.axvline(gv.mean(self.phys_point_data['m_'+yparam]), ls='--', label='phys. point')

    #     fig = plt.gcf()
    #     plt.close()
    #     return fig


    @property
    def hyp_mass(self):
        return self.fitfcn(data=self.phys_point_data.copy())

    def _extrapolate_to_ens(self, ens=None, phys_params=None):
        if phys_params is None:
            phys_params = []

        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xdata = {}
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fitter.fit.p:
                    shape = self.fitter.fit.p[param].shape
                    if param in phys_params:
                        posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc']
                    elif shape == ():
                        posterior[param] = self.fitter.fit.p[param]
                    else:
                        posterior[param] = self.fitter.fit.p[param][j]

                if 'alpha_s' in phys_params:
                    posterior['alpha_s'] = self.phys_point_data['alpha_s']

                if 'eps_pi' in phys_params:
                    xdata['eps_pi'] = self.phys_point_data['m_pi'] / self.phys_point_data['lam_chi']
                if 'd_eps2_s' in phys_params:
                    xdata['d_eps2_s'] = (2 *self.phys_point_data['m_k']**2 - self.phys_point_data['m_pi']**2)/ self.phys_point_data['lam_chi']**2
                if 'eps_a' in phys_params:
                    xdata['eps_a'] = 0

                if ens is not None:
                    return self.fitfcn(p=posterior, data={}, particle=None)
                else:
                    extrapolated_values[ens_j] = self.fitfcn(p=posterior, data={}, particle=None)

                
            extrapolated_values[ens_j] = self.fitfcn(p=posterior, data={}, particle=None)
        return extrapolated_values

    def shift_latt_to_phys(self, ens=None, phys_params=None):
        value_shifted = {}
        for j, ens_j in enumerate(self.ensembles):
            if ens is None or ens_j == ens:
                value_latt = self.fit.y.values()[0][j]
                value_fit = self._extrapolate_to_ens(ens=j)
                value_fit_phys = self._extrapolate_to_ens(ens_j, phys_params)

                value_shifted[ens_j] = value_latt + value_fit_phys - value_fit
                if ens is not None:
                    return value_shifted[ens_j]

        return value_shifted