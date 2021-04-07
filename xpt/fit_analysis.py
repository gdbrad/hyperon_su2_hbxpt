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
from fit_routine import fit_routine as fit
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
        self.fit = fit
        #self.fitter = {} # fill fitter dict with models based on scheme, use 'wo_imp' for now 
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = {}

    # def __str__(self):
    #     output = "Model: %s" %(self.model)

    #     output += '\nParameters:\n'
    #     my_str = str(fitter.fit_routine)
    #     for item in my_str.split('\n'):
    #         for key in self.fit_keys:
    #             re = key+' '
    #             if re in item:
    #                 output += item + '\n'
    #     return output

    
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
        return self.model_info['particles']

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
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fit.p:
                    shape = self.fit.p[param].shape
                    if param in phys_params:
                        posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc']
                    elif shape == ():
                        posterior[param] = self.fit.p[param]
                    else:
                        posterior[param] = self.fit.p[param][j]
            extrapolated_values[ens_j] = self.fitfcn(posterior=posterior, data={}, particle=None)
        return extrapolated_values

    def fitfcn(self, posterior=None, data=None, particle=None):
        output = {}
        for mdl in self.fit._make_models():
            part = mdl.datatag
            output[part] = mdl.fitfcn(p=posterior,data)

        if particle is None:
            return output
        else:
            return output[particle]

    def plot_parameters(self, xparam, yparam=None):
        if yparam is None:
            yparam = 'w0mO'

        x = {}
        y = {}
        c = {}
            
        colors = {
            '06' : '#6A5ACD',#'#00FFFF',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }
        #c = {abbr : colors[ens[1:3]] for abbr in self.ensembles}

        for ens in self.ensembles:
            for j, param in enumerate([xparam, yparam]):
                if param == 'eps_pi':
                    value = (self.fit_data[ens]['mpi']**2 / self.fit_data[ens]['lam_chi'])**2
                    label= r'$\xi_l$'
                elif param == 'sf':
                    value = ((2 *self.fit_data[ens]['m_k'] - self.fit_data[ens]['m_pi'])/ self.fit_data[ens]['lam_chi'])**2
                    label = r'$\xi_s$'
                elif param == 'eps_a':
                    value = self.fit_data[ens]['a/w']**2 / 4
                    label = r'$\xi_a$'
                elif param == 'm_pi':
                    value = self.fit_data[ens]['m_pi']
                    label = '$am_\pi$'


                if j == 0:
                    x[ens] = value
                    xlabel = label
                elif j == 1:
                    y[ens] = value
                    ylabel = label

        for ens in self.ensembles:
            C = gv.evalcov([x[ens], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[ens])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[ens])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        color=colors[ens[1:3]], alpha=1.0, lw=2)
                plt.plot(gv.mean(x[ens]), gv.mean(y[ens]), 
                         color=colors[ens[1:3]], marker='o', mec='w', zorder=3)


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
            ncol=len(by_label), bbox_to_anchor=(0,1), loc='lower left')
        plt.grid()
        plt.xlabel(xlabel, fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_fit(self, param, show_legend=True, ylim=None):
        x = {}
        y = {}
        c = {}

        plt.axes([0.145,0.145,0.85,0.85])

        colors = {
            '06' : '#6A5ACD',#'#00FFFF',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }

        if observable == 'w0':
            latt_spacings = {a_xx[1:] for a_xx in ['a06', 'a09' , 'a12', 'a15']}
        latt_spacings['00'] = gv.gvar(0,0)

        for j, xx in enumerate(reversed(latt_spacings)):
            xi = {}
            phys_data = self.phys_point_data
            phys_data['a'] = latt_spacings[xx]

            min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                       gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))


            if param in ['m_pi', 'eps_pi']:
                plt.axvline(gv.mean((phys_data['m_pi'] / phys_data['lam_chi'])**2), ls='--', label='phys. point')
                min_max = min_max({ens : (self.fit_data[ens]['m_pi'] / self.fit_data[ens]['lam_chi'])**2 for ens in self.ensembles})
                xi['eps_pi'] = np.linspace(0.0001, min_max[1])
                x_fit = xi['l']

            elif param in ['m_k', 'sf']:
                plt.axvline(gv.mean(((2 *phys_data['m_k']**2 - phys_data['m_pi']**2) / phys_data['lam_chi']**2)), ls='--', label='Phys point')
                min_max = min_max({ens : (2 *self.fit_data[ens]['m_k']**2 - self.fit_data[ens]['m_pi']**2) / self.fit_data[ens]['lam_chi']**2 for ens in self.ensembles})
                xi['sf'] = np.linspace(min_max[0], min_max[1])
                x_fit = xi['sf']

            elif param == 'eps_a':
                plt.axvline(0, label='phys. point', ls='--')
                
                if self.model_info['eps2a_defn'] == 'w0_imp':
                    eps2_a_arr = [self.fit_data[ens]['a/w']**2 / 4 for ens in self.ensembles] 
                xi['eps_a'] = np.linspace(0, gv.mean(np.max(eps2_a_arr)))
                x_fit = xi['eps_a']

                
            y_fit = self.fitfcn(posterior=self.posterior, fit_data=phys_data)

            # For LO fits
            if not hasattr(y_fit, "__len__"):
                y_fit = np.repeat(y_fit, len(x_fit))


            pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
            if xx != '00' and param != 'a':
                plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), color=colors[xx], alpha=0.4)
            elif xx == '00' and param != 'w0mO':
                plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), facecolor='None', edgecolor='k', alpha=0.6, hatch='/')
            else:
                pass

        for ens in self.ensembles:

            if param in ['pi', 'l', 'p']:
                x[ens] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_s', 'alpha_s'], observable=observable))
                label = r'$\xi_l$'
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    label = r'$l^2_F = m_\pi^2 / (4 \pi F_\pi)^2$'

            elif param in ['k', 's']:
                x[ens] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_l', 'alpha_s'], observable=observable))
                label = r'$\xi_s$'
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    label = r'$s^2_F$'

            elif param == 'a':
                if self.model_info['eps2a_defn'] == 'w0_original':
                    x[ens] = self.fit_data[ens]['a/w:orig']**2 / 4
                    label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{orig}})^2$'
                elif self.model_info['eps2a_defn'] == 'w0_improved':
                    x[ens] = self.fit_data[ens]['a/w:impr']**2 / 4
                    label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{impr}})^2$'
                elif self.model_info['eps2a_defn'] == 't0_original':
                    x[ens] = 1 / self.fit_data[ens]['t/a^2:orig'] / 4
                    label = r'$\epsilon^2_a = t_{0,\mathrm{orig}} / 4 a^2$'
                elif self.model_info['eps2a_defn'] == 't0_improved':
                    x[ens] = 1 / self.fit_data[ens]['t/a^2:impr'] / 4
                    label = r'$\epsilon^2_a = t_{0,\mathrm{impr}} / 4 a^2$'
                elif self.model_info['eps2a_defn'] == 'variable':
                    if observable == 'w0':
                        x[ens] = self.fit_data[ens]['a/w']**2 / 4
                        label = '$\epsilon^2_a = (a / 2 w_{0,\mathrm{var}})^2$'
                    elif observable == 't0':
                        x[ens] = 1 / self.fit_data[ens]['t/a^2'] / 4
                        label = '$\epsilon^2_a = t_{0,\mathrm{var}} / 4 a^2$'

                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_l', 'xi_s', 'alpha_s'], observable=observable))
                #label = '$\epsilon^2_a = (a / 2 w_0)^2$'

        for ens in reversed(self.ensembles):
            C = gv.evalcov([x[ens], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[ens])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[ens])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        color=colors[ens[1:3]], alpha=1.0, lw=2)
                plt.plot(gv.mean(x[ens]), gv.mean(y[ens]), 
                         color=colors[ens[1:3]], marker='o', mec='w', zorder=3)

        if show_legend:
            if param in ['pi', 'l', 'p']:
                labels = [
                    r'$a_{06}(l_F,s_F^{\rm phys})$',
                    r'$a_{09}(l_F,s_F^{\rm phys})$',
                    r'$a_{12}(l_F,s_F^{\rm phys})$',
                    r'$a_{15}(l_F,s_F^{\rm phys})$'
                ]
            elif param in ['k', 's']:
                labels = [
                    r'$a_{06}(l_F^{\rm phys},s_F)$',
                    r'$a_{09}(l_F^{\rm phys},s_F)$',
                    r'$a_{12}(l_F^{\rm phys},s_F)$',
                    r'$a_{15}(l_F^{\rm phys},s_F)$'
                ]
            elif param == 'a':
                labels = [
                    r'$a_{06}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{09}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{12}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{15}(l_F^{\rm phys},s_F^{\rm phys})$'
                ]
            handles = [
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['06']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['09']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['12']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['15'])
            ]
            plt.legend(handles=handles, labels=labels, ncol=2)#, bbox_to_anchor=(0,1), loc='lower left')

        #plt.grid()
        plt.xlabel(label)

        if observable == 'w0':
            plt.ylabel('$w_0 m_\Omega$')
        elif observable == 't0':
            plt.ylabel('$m_\Omega \sqrt{t_0 / a^2}$')

        if ylim is not None:
            plt.ylim(ylim)

        fig = plt.gcf()
        plt.close()
        return fig


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





