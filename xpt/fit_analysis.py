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
import fit_routine as fit
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
        self.fitter = {}
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = {}
        self.fitter = fit.fit_routine(prior=prior,data=data, model_info=model_info)

        # def __str__(self):
        #     output = "Model: %s" %(self.model) 
        #     output += '\nError Budget:\n'
        #     max_len = np.max([len(key) for key in self.error_budget[obs]])
        #     for key in {k: v for k, v in sorted(self.error_budget[obs].items(), key=lambda item: item[1], reverse=True)}:
        #         output += '  '
        #         output += key.ljust(max_len+1)
        #         output += '{: .1%}\n'.format((self.error_budget[obs][key]/self..sdev)**2).rjust(7)

            
        #     return output

    ## It is particularly useful for analyzing the impact of the a priori uncertainties encoded in the prior
    @property
    def error_budget(self):
        return self._get_error_budget()

    def _get_error_budget(self, **kwargs):
        
        #output = None

        #use above dict to fill in values where particle name goes.. leave hardcoded for now
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
        'm_{lambda,0}', 'm_{sigma,0}', 'm_{sigma_st,0}', 'm_{xi,0}', 'm_{xi_st,0}', 'd_{lambda,a}', 'd_{sigma,a}',  
        'd_{sigma_st,a}', 'd_{xi,a}',  'd_{xi_st,a}', 'd_{lambda,aa}', 'd_{lambda,al}', 
        'd_{sigma,aa}', 'd_{sigma,al}',  'd_{sigma_st,aa}', 'd_{sigma_st,al}', 
        'd_{xi,aa}', 'd_{xi,al}',  'd_{xi_st,aa}', 'd_{xi_st,al}']
        
        phys_keys = list(self.phys_point_data)
        stat_keys = ['lam_chi','eps2_a','m_lambda','m_pi','m_k']
        

        # if verbose:
        #     if output is None:
        #         output = ''

        inputs = {}

        # # xpt/chiral contributions
        # inputs.update({str(param)+' [disc]' : self._input_prior[param] for param in disc_keys if param in self._input_prior})
        # inputs.update({str(param)+' [xpt]' : self._input_prior[param] for param in chiral_keys if param in self._input_prior})
        # inputs.update({str(param)+ '[strange]' : self._input_prior[param] for param in strange_keys if param in self._input_prior})

        # # phys point contributions
        # inputs.update({str(param)+' [pp]' : self.phys_point_data[param] for param in list(phys_keys)})

        inputs.update({str(param): self._input_prior[param] for param in disc_keys if param in self._input_prior})
        inputs.update({str(param): self._input_prior[param] for param in chiral_keys if param in self._input_prior})
        inputs.update({str(param): self._input_prior[param] for param in strange_keys if param in self._input_prior})

        # phys point contributions
        inputs.update({str(param): self.phys_point_data[param] for param in list(phys_keys)})
        #del inputs['lam_chi [pp]']

        #stat contribtions
        inputs.update({'x [stat]' : self._input_prior[param] for param in stat_keys if param in self._input_prior})# , 'y [stat]' : self.fitter.fit.y})
        print(inputs.values())

        if kwargs is None:
            kwargs = {}
        kwargs.setdefault('percent', False)
        kwargs.setdefault('ndecimal', 10)
        kwargs.setdefault('verify', True)
        
        #output = {}

        #output = {}
        extrapolated_mass = {}
        for particle in self.model_info['particles']:
            extrapolated_mass[particle] = self.fitfcn(p=self.posterior, data=self.phys_point_data, particle=particle)
        #print(inputs.keys())
        #print(extrapolated_mass)

        #value = extrapolated_mass.partialsdev([self.prior[key] for key in disc_keys if key in self.prior])
        #for keys in disc_keys:
        print(gv.fmt_errorbudget(outputs=extrapolated_mass, inputs=inputs, verify=True))
        output = {}

        output['disc'] = extrapolated_mass[particle].partialsdev(
                    [self.prior[particle] for key in disc_keys if key in self.prior])
        output['pp'] = extrapolated_mass[particle].partialsdev(
                    [self.prior for key in phys_keys if key in self.prior])

        output['stat'] = extrapolated_mass[particle].partialsdev(
                    [self.prior for key in stat_keys if key in self.prior])

        output['chiral'] = extrapolated_mass[particle].partialsdev(
                    [self.prior for key in chiral_keys if key in self.prior])
        
        
        print(output)
        #         #elif observable == 't0':
        # #output += 'observable: ' + observable + '\n' + gv.fmt_errorbudget(outputs={'t0' : self.sqrt_t0}, inputs=inputs, **kwargs) + '\n---\n'

        #print(value)

        # output['disc'] = value.partialsdev([self.prior[key] for key in disc_keys if key in self.prior]
        # )
        # output['chiral'] = value.partialsdev([self.prior[key] for key in chiral_keys if key in self.prior]
        # )
        # output['strange'] = value.partialsdev([self.prior[key] for key in strange_keys if key in self.prior]
        # )
        # output['pp_input'] = value.partialsdev([self.phys_point_data[key] for key in phys_keys]
        # )
        # output['stat'] = value.partialsdev([self.prior[key] for key in stat_keys if key in self.prior]
        # )

        

        # #     #output += '\n' + gv.fmt_errorbudget(outputs=outputs, inputs=inputs, **kwargs) + '\n---\n'
        # #     # elif== 't0':
        # #     #     output +=  ' ++ '\n' + gv.fmt_errorbudget(outputs={'t0' : self.sqrt_t0}, inputs=inputs, **kwargs) + '\n---\n'



        #return output

    # @property
    # def fit_info(self):
    #     #fit_info = {}
    #     fit_info = {
    #         'name' : self.model,
    #         #'w0_imp' : self.w0,
    #         'logGBF' : self.fitter.logGBF,
    #         'chi2/df' : self.fitter.chi2 / self.fitter.dof,
    #         'Q' : self.fit.Q,
    #         'phys_point' : self.phys_point_data,
    #         #'error_budget' : self.error_budget['w0'],
    #         'prior' : self.prior,
    #         'posterior' : self.posterior
    #     }
    #     return fit_info

    # Returns names of LECs in prior/posterior
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
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self._phys_point_data
        else:
            return self._phys_point_data[parameter]

    @property
    def posterior(self):
        return self._get_posterior()

    # # Returns dictionary with keys fit parameters, entries gvar results

    def _get_posterior(self,param=None):
        #output = {}
        #return self.fit.p
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
            output = {param : self.fitter.fit.prior}
        elif param == 'all':
            output = self.fitter.fit.prior
        else:
            output = self.fitter.fit.prior[param]

        return output

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

    def fitfcn(self, p, data=None, particle=None):
        output = {}
        # if p is None:
        #     p = self.posterior

        for mdl in self.fitter._make_models():
            part = mdl.datatag
            output[part] = mdl.fitfcn(p,data)

        if particle is None:
            return output
        else:
            return output[particle]

    # Takes keys from posterior (eg, 'A_l' and 'A_s')
    # def plot_error_ellipsis(self, x_key, y_key):
    #     x = self._get_posterior(x_key)
    #     y = self._get_posterior(y_key)


    #     fig, ax = plt.subplots()

    #     corr = '{0:.3g}'.format(gv.evalcorr([x, y])[0,1])
    #     std_x = '{0:.3g}'.format(gv.sdev(x))
    #     std_y = '{0:.3g}'.format(gv.sdev(y))
    #     text = ('$R_{x, y}=$ %s\n $\sigma_x =$ %s\n $\sigma_y =$ %s' %(corr,std_x,std_y))

    #     # these are matplotlib.patch.Patch properties
    #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    #     # place a text box in upper left in axes coords
    #     ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
    #             verticalalignment='top', bbox=props)

    #     C = gv.evalcov([x, y])
    #     eVe, eVa = np.linalg.eig(C)
    #     for e, v in zip(eVe, eVa.T):
    #         plt.plot([gv.mean(x)-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x)],
    #                  [gv.mean(y)-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y)],
    #                  'k-', lw=2)

    #     #plt.scatter(x-np.mean(x), y-np.mean(y), rasterized=True, marker=".", alpha=100.0/self.bs_N)
    #     #plt.scatter(x, y, rasterized=True, marker=".", alpha=100.0/self.bs_N)

    #     plt.grid()
    #     plt.gca().set_aspect('equal', adjustable='datalim')
    #     plt.xlabel(x_key.replace('_', '\_'), fontsize = 24)
    #     plt.ylabel(y_key.replace('_', '\_'), fontsize = 24)

    #     fig = plt.gcf()
    #     plt.close()
    #     return fig


    
    def plot_fit(self, param, show_legend=True, ylim=None,):
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
        #ensembles = [ens for ens in self.ensembles if ens[:3] == latt_spacing]

        
        #latt_spacings = {a_xx[1:] for a_xx in ['a06', 'a09' , 'a12', 'a15']}
        #latt_spacings['00'] = gv.gvar(0,0)

        for j, xx in enumerate(latt_spacings):
            xdata = {}
            phys_data = self.phys_point_data
            phys_data['eps2_a'] = latt_spacings[xx]

            min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                       gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))


            if param in ['m_pi', 'eps_pi', 'p']:
                plt.axvline(gv.mean((phys_data['m_pi'] / phys_data['lam_chi'])**2), ls='--', label='phys. point')
                min_max = min_max({ens : (self.data[ens]['m_pi'] / self.data[ens]['lam_chi'])**2 for ens in self.ensembles})
                xdata['eps_pi'] = np.linspace(0.0001, min_max[1])
                x_fit = xdata['eps_pi']

            elif param in ['m_k', 'd_eps2_s']:
                plt.axvline(gv.mean(((2 *phys_data['m_k']**2 - phys_data['m_pi']**2) / phys_data['lam_chi']**2)), ls='--', label='Phys point')
                min_max = min_max({ens : (2 *self.data[ens]['m_k']**2 - self.data[ens]['m_pi']**2) / self.data[ens]['lam_chi']**2 for ens in self.ensembles})
                xdata['d_eps2_s'] = np.linspace(min_max[0], min_max[1])
                x_fit = xdata['d_eps2_s']

            elif param == 'eps_a':
                plt.axvline(0, label='phys. point', ls='--')
                
                eps2_a_arr = [self.data[ens]['a/w']**2 / 4 for ens in self.ensembles] 
                xdata['eps_a'] = np.linspace(0, gv.mean(np.max(eps2_a_arr)))
                x_fit = xdata['eps_a']

                
            y_fit = self.fitfcn(p=self.posterior, data=phys_data, particle=None)

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

            if param in ['m_pi', 'eps_pi', 'p']:
                x[ens] = (self.data[ens]['m_pi'] / self.data[ens]['lam_chi'])**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['d_eps2_s', 'alpha_s']))
                label = r'$\xi_l$'
                # if self.model_info['chiral_cutoff'] == 'Fpi':
                #     label = r'$l^2_F = m_\pi^2 / (4 \pi F_\pi)^2$'

            elif param in ['m_k', 'd_eps2_s']:
                x[ens] = (2 *self.data[ens]['m_k']**2 - self.data[ens]['m_pi']**2) / self.data[ens]['lam_chi']**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['eps_pi', 'alpha_s']))
                label = r'$\xi_s$'
                # if self.model_info['chiral_cutoff'] == 'Fpi':
                #     label = r'$s^2_F$'

            elif param == 'eps_a':
                x[ens] = self.data[ens]['a/w']**2 / 4
                label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{impr}})^2$'
                # if self.model_info['eps2a_defn'] == 'w0_original':
                #     x[ens] = self.fit_data[ens]['a/w']**2 / 4
                #     label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{orig}})^2$'
                # elif self.model_info['eps2a_defn'] == 'w0_improved':
                #     x[ens] = self.fit_data[ens]['a/w:impr']**2 / 4
                #     label = r'$\epsilon^2_a = (a / 2 w_{0,\mathrm{impr}})^2$'
                # elif self.model_info['eps2a_defn'] == 't0_original':
                #     x[ens] = 1 / self.fit_data[ens]['t/a^2:orig'] / 4
                #     label = r'$\epsilon^2_a = t_{0,\mathrm{orig}} / 4 a^2$'
                # elif self.model_info['eps2a_defn'] == 't0_improved':
                #     x[ens] = 1 / self.fit_data[ens]['t/a^2:impr'] / 4
                #     label = r'$\epsilon^2_a = t_{0,\mathrm{impr}} / 4 a^2$'
                # elif self.model_info['eps2a_defn'] == 'variable':
                #     if== 'w0':
                #         x[ens] = self.fit_data[ens]['a/w']**2 / 4
                #         label = '$\epsilon^2_a = (a / 2 w_{0,\mathrm{var}})^2$'
                #     elif== 't0':
                #         x[ens] = 1 / self.fit_data[ens]['t/a^2'] / 4
                #         label = '$\epsilon^2_a = t_{0,\mathrm{var}} / 4 a^2$'

                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['eps_pi', 'd_eps2_s', 'alpha_s']))
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
            if param in ['eps_pi', 'd_eps2_s', 'p']:
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
        plt.ylabel('$w_0_imp$')

        # if== 'w0':
        #     plt.ylabel('$w_0 m_\Omega$')
        # elif== 't0':
        #     plt.ylabel('$m_\Omega \sqrt{t_0 / a^2}$')

        if ylim is not None:
            plt.ylim(ylim)

        fig = plt.gcf()
        plt.close()
        return fig

    def plot_parameters(self, xparam, yparam=None):
        # if yparam is None:
        #     yparam = 'w0mO'

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
                    value = (self.data[ens]['m_pi'] / self.data[ens]['lam_chi'])
                    label= r'$\eps_pi$'
                elif param == 'd_eps2_s':
                    value = ((2 *self.data[ens]['m_k'] - self.data[ens]['m_pi'])/ self.data[ens]['lam_chi'])**2
                    label = r'$d_\eps2s$'
                elif param == 'eps2_a':
                    value = self.data[ens]['eps2_a']
                    label = r'$\eps2_a$'
                elif param == 'm_pi':
                    value = self.data[ens]['m_pi']
                    label = '$m_\pi$'


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