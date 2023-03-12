import time
import sys
import lsqfit
import os
import pandas as pd
import numpy as np
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import cmath
from gmo_fitter import fitter
import mass_relations as ma 


class fit_analysis(object):

    def __init__(self, t_range,t_period, prior,p_dict, n_states=None, model_type = None,states=None,simult=None,gmo_type=None,
                 nucleon_corr_data=None,lam_corr_data=None,
                 xi_corr_data=None,sigma_corr_data=None,
                 gmo_corr_data = None
                 ):
        #All fit ensembles (manual and automatic) must have these variables
        
        #Convert correlator data into gvar dictionaries
        if nucleon_corr_data is not None:
            nucleon_corr_gv = gv.dataset.avg_data(nucleon_corr_data)
        else:
            nucleon_corr_gv = None
        if lam_corr_data is not None:
            lam_corr_gv = gv.dataset.avg_data(lam_corr_data)
        else:
            lam_corr_gv = None
        if xi_corr_data is not None:
            xi_corr_gv = gv.dataset.avg_data(xi_corr_data)
        else:
            xi_corr_gv = None
        if sigma_corr_data is not None:
            sigma_corr_gv = gv.dataset.avg_data(sigma_corr_data)
        else:
            sigma_corr_gv = None
        if gmo_corr_data is not None:
            gmo_corr_gv =gmo_corr_data
        else:
            gmo_corr_gv = None

        # Default to a 1 state fit
        if n_states is None:
            n_states = 1

        for data_gv in [nucleon_corr_gv]:
            if data_gv is not None:
                t_max = len(data_gv[list(data_gv.keys())[0]])


        t_start = np.min([t_range[key][0] for key in list(t_range.keys())])
        t_end = np.max([t_range[key][1] for key in list(t_range.keys())])

        max_n_states = np.max([n_states[key] for key in list(n_states.keys())])
        self.p_dict = p_dict
        self.model_type = model_type
        self.nucleon_corr_gv = nucleon_corr_gv
        self.lam_corr_gv = lam_corr_gv
        self.sigma_corr_gv = sigma_corr_gv
        self.xi_corr_gv = xi_corr_gv
        self.gmo_corr_gv = gmo_corr_gv
        self.states=states 
        self.gmo_type = gmo_type
        self.simult = simult
        self.n_states = n_states
        self.prior = prior
        self.t_range = t_range
        self.t_period = t_period
        self.t_delta = 2*max_n_states
        self.t_min = int(t_start/2)
        self.t_max = int(np.min([t_end, t_end]))
        self.fits = {}
        
        #self.bs = None

    # def __str__(self):
    #     output = "Model: %s" %(self.model)
    #     for obs in ['z_gmo', 'z_baryons','d_gmo','d_z_gmo','4_baryon']:
    #         output += '\n---\n'

    #         if obs == 'w0':
    #             output += "\nw0: %s\n\n" %(self.w0)
    #             for a_xx in ['a06', 'a09', 'a12', 'a15']:
    #                 w0_a = self.interpolate_w0a(latt_spacing=a_xx)
    #                 output += 'w0/{}: {}'.format(a_xx, w0_a).ljust(22)  + '=> %s/fm: %s\n'%(a_xx, self.w0 / w0_a)

    #         elif obs == 't0':
    #             output += "\nsqrt(t0): %s\n\n" %(self.sqrt_t0)
    #             for a_xx in ['a06', 'a09', 'a12', 'a15']:
    #                 t0_a2 = self.interpolate_t0a2(latt_spacing=a_xx)
    #                 output += 't0/{}^2: {}'.format(a_xx, t0_a2).ljust(22)  + '=> %s/fm: %s\n'%(a_xx, self.sqrt_t0 / np.sqrt(t0_a2))



    #         output += '\nParameters:\n'
    #         my_str = self.fit[obs].format(pstyle='m')
    #         for item in my_str.split('\n'):
    #             for key in self.fit_keys['w0']:
    #                 re = key+' '
    #                 if re in item:
    #                     output += item + '\n'

    #         output += '\n'
    #         output += self.fit[obs].format(pstyle=None)

    @property
    def posterior(self):
        return self._get_posterior()

    # Returns dictionary with keys fit parameters, entries gvar results
    def _get_posterior(self, param=all):
        output = {}
        for observable in ['z_gmo','d_gmo','d_z_gmo','4_baryon']:
            if param is None:
                output[observable] = {param : self.get_fit[observable].p[param] for param in self.fit_keys[observable]}
            elif param == 'all':
                output[observable] = self.get_fit[observable].p
            else:
                output[observable] = self.get_fit[observable].p[param]

        return output

    # def fit(self):
    #     for observable in ['z_gmo','d_gmo','d_z_gmo','4_baryon']:
    #         self.fit[observable] =self.get_fit[observable]
    #     return self.fit

    def get_fit(self, t_range=None, n_states=None,t_period=None):
        if t_range is None:
            t_range = self.t_range
        if n_states is None:
            n_states = self.n_states
        if t_period is None:
            t_period = self.t_period

        index = tuple((t_range[key][0], t_range[key][1], n_states[key]) for key in sorted(t_range.keys()))
        # print(index)

        temp_fit = {}
        if index in list(self.fits.keys()):
            return self.fits[index]
        else:
            for gmo_type in ['z_gmo','d_gmo','d_z_gmo','4_baryon']:
                temp_fit[gmo_type] = fitter(n_states=n_states,prior=self.prior,p_dict=self.p_dict, t_range=t_range,states=self.states,simult=self.simult, t_period=t_period,model_type=self.model_type,gmo_type=gmo_type,
                               nucleon_corr=self.nucleon_corr_gv,lam_corr=self.lam_corr_gv,
                               xi_corr=self.xi_corr_gv,sigma_corr=self.sigma_corr_gv,
                               gmo_ratio_corr=self.gmo_corr_gv).get_fit()

                self.fits[gmo_type] = temp_fit[gmo_type]
        return temp_fit

    # type should be either "corr, "gA", or "gV"
    def _get_models(self, model_type=None):
        
        # data_gv = [nucleon_corr , lam_corr ,sigma_corr, xi_corr]?
        if model_type is None:
            return None
        #data_gv = [nucleon_corr , lam_corr ,sigma_corr, xi_corr] 
        elif model_type == "sigma":
            nucleon_corr = None
            lam_corr = None
            sigma_corr = self.sigma_corr_gv
            xi_corr = None
            gmo_ratio_corr = None


        elif model_type == "xi":
            nucleon_corr = None
            lam_corr = None
            sigma_corr = None
            xi_corr = self.xi_corr_gv
            gmo_ratio_corr = None


        elif model_type == "lam":
            nucleon_corr = None
            lam_corr = self.lam_corr_gv
            sigma_corr = None
            xi_corr = None
            gmo_ratio_corr = None

        elif model_type == "proton":
            nucleon_corr = self.nucleon_corr_gv
            lam_corr = None
            sigma_corr = None
            xi_corr = None
            gmo_ratio_corr = None

        elif model_type == "simult_baryons":
            nucleon_corr = self.nucleon_corr_gv
            lam_corr = self.lam_corr_gv
            sigma_corr = self.sigma_corr_gv
            xi_corr = self.xi_corr_gv
            gmo_ratio_corr = None

        elif model_type == "gmo_direct":
            nucleon_corr = None
            lam_corr = None
            sigma_corr = None
            xi_corr = None
            gmo_ratio_corr = self.gmo_corr_gv

        elif model_type == "simult_baryons_gmo":
            nucleon_corr = self.nucleon_corr_gv
            lam_corr = self.lam_corr_gv
            sigma_corr = self.sigma_corr_gv
            xi_corr = self.xi_corr_gv
            gmo_ratio_corr = self.gmo_corr_gv

        elif model_type == "simult_gmo_linear":
            nucleon_corr = self.nucleon_corr_gv
            lam_corr = self.lam_corr_gv
            sigma_corr = self.sigma_corr_gv
            xi_corr = self.xi_corr_gv
            gmo_ratio_corr = self.gmo_corr_gv


        else:
            return None 

        return fitter(n_states=self.n_states,gmo_type=self.gmo_type, states=self.states,prior=self.prior,simult=self.simult, p_dict=self.p_dict, 
                        t_range=self.t_range,t_period=self.t_period,model_type=self.model_type,
                      nucleon_corr=nucleon_corr,lam_corr=lam_corr,xi_corr=xi_corr,sigma_corr=sigma_corr,gmo_ratio_corr=gmo_ratio_corr)._make_models_simult_fit()

    def _generate_data_from_fit(self, t, t_start=None, t_end=None, model_type=None, n_states=None):
        if model_type is None:
            return None

        if t_start is None:
            t_start = self.t_range[model_type][0]
        if t_end is None:
            t_end = self.t_range[model_type][1]
        if n_states is None:
            n_states = self.n_states

        # Make
        t_range = {key : self.t_range[key] for key in list(self.t_range.keys())}
        t_range[model_type] = [t_start, t_end]

        models = self._get_models(model_type=model_type)
        fit = self.get_fit(t_range=t_range, n_states=n_states)

        # datatag[-3:] converts, eg, 'nucleon_dir' -> 'dir'
        output = {model.datatag : model.fitfcn(p=fit.p, t=t) for model in models}
        return output

    def get_gmo_effective(self, gmo_ratio=None, t=None, dt=None):
        if gmo_ratio is None:
            gmo_ratio = self.gmo_corr_gv

        # If still empty, return nothing
        if gmo_ratio is None:
            return None

        if dt is None:
            dt = 1
        return {key : 1/dt * np.log(gmo_ratio[key] / np.roll(gmo_ratio[key], -1))
                for key in list(gmo_ratio.keys())}

    # def plot_m4(self,correlators_gv=None,model_type=None, t_plot_min = None, t_plot_max = None,fig_name=None,show_fit=None):

    def plot_gmo_effective_mass(self, effective_mass,t_plot_min=None, model_type=None,
                            t_plot_max=None, show_plot=True, show_fit=True,fig_name=None):
        if t_plot_min is None:
            t_plot_min = self.t_min
        if t_plot_max is None:
            t_plot_max = self.t_max
        markers = ["^", "v"]
        colors = np.array(['red', 'blue','green','magenta'])
        t = np.arange(t_plot_min, t_plot_max)
        # effective_mass = {}
        if model_type == None:
            raise TypeError(model_type,'you need to supply a correlator model in order to generate an eff mass plot for that correlator')
        if effective_mass is None:
            return None

        y = {}
        y_err = {}
        lower_quantile = np.inf
        upper_quantile = -np.inf
        for j, key in enumerate(effective_mass.keys()):
            y[key] = gv.mean(effective_mass[key])[t]
            y_err[key] = gv.sdev(effective_mass[key])[t]
            lower_quantile = np.min([np.nanpercentile(y[key], 25), lower_quantile])
            upper_quantile = np.max([np.nanpercentile(y[key], 75), upper_quantile])

            plt.errorbar(x=t, y=y[key], xerr=0.0, yerr=y_err[key], fmt='o', capsize=2.0,
                color = colors[j%len(colors)], capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)
        delta_quantile = upper_quantile - lower_quantile
        plt.ylim(lower_quantile - 0.5*delta_quantile,
                 upper_quantile + 0.5*delta_quantile)

        if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = self._generate_data_from_fit(model_type=model_type, t=t)

            for j, key in enumerate(fit_data_gv.keys()):
                eff_mass_fit = self.get_nucleon_effective_mass(fit_data_gv, dt)[key][1:-1]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(eff_mass_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(eff_mass_fit, 1), t[1:-1], pm(eff_mass_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(eff_mass_fit, -1), pm(eff_mass_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("gmo_ratio_eff for $N_{states} = $%s" %(self.n_states['gmo']), fontsize = 24)

        plt.xlim(t_plot_min-0.5, t_plot_max-.5)
        # plt.ylim(0.0,0.03)
         # Get unique markers when making legend
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {}
        for j, handle in enumerate(handles):
            temp[labels[j]] = handle

        plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])
        # plt.legend()
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel('$M^{eff}_{GMO}$', fontsize = 24)
        fig = plt.gcf()
        # plt.savefig(fig_name)
        # if show_plot == True: plt.show()
        # else: plt.close()
        plt.close()

        return fig

    def plot_delta_gmo(self,correlators_gv=None,model_type=None, t_plot_min = None, 
    t_plot_max = None,fig_name=None,show_fit=None):
        if t_plot_min == None: t_plot_min = 0
        # if t_plot_max == None: t_plot_max = correlators_gv[correlators_gv.keys()[0]].shape[0] - 1
        colors = np.array(['red', 'blue', 'green','magenta'])

        x = np.arange(t_plot_min,t_plot_max)
        # t = {}

        y = {}
        y_err = {}
        lower_quantile = np.inf
        upper_quantile = -np.inf
        for j, key in enumerate(sorted(correlators_gv.keys())):
            # t[key] = np.arange(t_plot_min, t_plot_max)

            pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
            y[key] = gv.mean(correlators_gv[key])[x]
            y_err[key] = gv.sdev(correlators_gv[key])[x]

            lower_quantile = np.min([np.nanpercentile(y[key], 25), lower_quantile])
            upper_quantile = np.max([np.nanpercentile(y[key], 75), upper_quantile])
            
            plt.errorbar(x=x, y=y[key], xerr = 0.0, yerr=y_err[key], fmt='o', capsize=5.0,
            color = colors[j%len(colors)],capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)
        delta_quantile = upper_quantile - lower_quantile
        plt.ylim(lower_quantile - 0.5*delta_quantile,
                 upper_quantile + 0.5*delta_quantile)
            # Label dirac/smeared data
            # plt.legend()
            # plt.grid(True)
            # plt.xlabel('$t$', fontsize = 24)
            # plt.ylabel('$G^{GMO}(t)$', fontsize = 24)

        if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = self._generate_data_from_fit(model_type=model_type, t=t)
            gmo_fit_data = {}
            gmo_fit_data['SS'] = fit_data_gv['gmo_ratio_SS']
            gmo_fit_data['PS'] = fit_data_gv['gmo_ratio_PS']

            t = t[1:-1]

            for j, key in enumerate(sorted(gmo_fit_data.keys())):
                A_eff_fit = gmo_fit_data[key][2:-2]
                # print(A_eff_fit)
                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(A_eff_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(A_eff_fit, 1), t[1:-1], pm(A_eff_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(A_eff_fit, -1), pm(A_eff_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("Best fit for $N_{states} = $%s" %(self.n_states['simult_baryons_gmo']), fontsize = 24)
        
        plt.xlim(t_plot_min-0.5, t_plot_max-.5)
        # plt.ylim(0.5,0.7)

        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel('$G_{GMO}(t)$')
        plt.grid(True)
        plt.legend()
        # if show_plot == True: plt.show() 
        # else: plt.close()

        fig = plt.gcf()
        plt.close()
        return fig

    def get_m_4(self,t=None):
        # output = {}
        if t is None:
            t = {key : np.arange(len(self.nucleon_corr_gv[key])) for key in list(self.nucleon_corr_gv.keys())}
        output= {key: self.nucleon_corr_gv[key] + self.lam_corr_gv[key] - 3*self.sigma_corr_gv[key] + self.xi_corr_gv[key]*t[key] for key in list(self.nucleon_corr_gv.keys())}
        # if mev:
            # output[self.abbr] = 197.3269804 * output[self.abbr] # MeV-fm
            # output = output 
        return output

    def get_nucleon_effective_mass(self, nucleon_corr_gv=None, dt=None):
        if nucleon_corr_gv is None:
            nucleon_corr_gv = self.nucleon_corr_gv

        # If still empty, return nothing
        if nucleon_corr_gv is None:
            return None

        if dt is None:
            dt = 1
        return {key : 1/dt * np.log(nucleon_corr_gv[key] / np.roll(nucleon_corr_gv[key], -1))
                for key in list(nucleon_corr_gv.keys())}

    def get_nucleon_effective_wf(self, nucleon_corr_gv=None, t=None, dt=None):
        if nucleon_corr_gv is None:
            nucleon_corr_gv = self.nucleon_corr_gv

        # If still empty, return nothing
        if nucleon_corr_gv is None:
            return None

        effective_mass = self.get_nucleon_effective_mass(nucleon_corr_gv, dt)
        if t is None:
            t = {key : np.arange(len(nucleon_corr_gv[key])) for key in list(nucleon_corr_gv.keys())}
        else:
            t = {key : t for key in list(nucleon_corr_gv.keys())}

        return {key : np.exp(effective_mass[key]*t[key]) * nucleon_corr_gv[key]
                for key in list(nucleon_corr_gv.keys())}

    def plot_effective_wf(self, model_type = None,t_plot_min=None,
                           t_plot_max=None, show_plot=True, show_fit=True):
        if t_plot_min is None:
            t_plot_min = self.t_min
        if t_plot_max is None:
            t_plot_max = self.t_max

        markers = ["^", "v"]
        colors = np.array(['red', 'blue', 'green','magenta','yellow','purple','cyan','teal'])
        t = np.arange(t_plot_min, t_plot_max)
        effective_wf = {}
        if model_type == None:
            raise TypeError(model_type,'you need to supply a correlator model in order to generate an eff mass plot for that correlator')
        elif model_type == 'simult_baryons':
            nucleon_corr_gv = self.nucleon_corr_gv
            xi_corr_gv      = self.xi_corr_gv
            lam_corr_gv     = self.lam_corr_gv
            sigma_corr_gv   = self.sigma_corr_gv
        elif model_type == 'xi':
            nucleon_corr_gv = None
            xi_corr_gv      = self.xi_corr_gv
            lam_corr_gv     = None
            sigma_corr_gv   = None
        elif model_type == 'gmo_ratio':
            nucleon_corr_gv = self.gmo_corr_gv
        
        effective_wf['proton'] = self.get_nucleon_effective_wf(nucleon_corr_gv)
        effective_wf['xi']     = self.get_nucleon_effective_wf(xi_corr_gv)
        effective_wf['sigma']  = self.get_nucleon_effective_wf(sigma_corr_gv)
        effective_wf['lam']    = self.get_nucleon_effective_wf(lam_corr_gv)

        if effective_wf is None:
            return None

        y = {}
        y_err = {}
        lower_quantile = np.inf
        upper_quantile = -np.inf
        for i,baryon in enumerate(effective_wf.keys()):
            y[baryon] = {}
            y_err[baryon] = {}
            for j, key in enumerate(effective_wf[baryon].keys()):
                y[baryon][key] = gv.mean(effective_wf[baryon][key])[t]
                y_err[baryon][key] = gv.sdev(effective_wf[baryon][key])[t]
                lower_quantile = np.min([np.nanpercentile(y[baryon][key], 25), lower_quantile])
                upper_quantile = np.max([np.nanpercentile(y[baryon][key], 75), upper_quantile])

                plt.errorbar(x=t, y=y[baryon][key], xerr=0.0, yerr=y_err[baryon][key], fmt=markers[j%len(markers)], capsize=4.0,
                    color = colors[i%len(colors)], capthick=1.0, alpha=0.6, elinewidth=5.0, label=baryon+'_'+key)
            delta_quantile = upper_quantile - lower_quantile
            plt.ylim(lower_quantile - 0.5*delta_quantile,
                    upper_quantile + 0.5*delta_quantile)

        if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = self._generate_data_from_fit(model_type=model_type, t=t)

            for j, key in enumerate(fit_data_gv.keys()):
                eff_mass_fit = self.get_nucleon_effective_wf(fit_data_gv, dt)[key][1:-1]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(eff_mass_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(eff_mass_fit, 1), t[1:-1], pm(eff_mass_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(eff_mass_fit, -1), pm(eff_mass_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("Simultaneous fit to 4 baryons for $N_{states} = $%s" %(self.n_states['gmo']), fontsize = 24)

        plt.xlim(t_plot_min-0.5, t_plot_max-.5)
        # plt.ylim(0.55,0.9)
         # Get unique markers when making legend
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {}
        for j, handle in enumerate(handles):
            temp[labels[j]] = handle

        plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])
        # plt.legend()
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel('$M^{eff}_{baryon}$', fontsize = 24)
        fig = plt.gcf()
        # plt.savefig(fig_name)
        if show_plot == True: plt.show()
        else: plt.close()
        return fig

    def plot_effective_mass(self, t_plot_min=None, model_type=None,
                            t_plot_max=None, show_plot=True, show_fit=True,fig_name=None):
        if t_plot_min is None:
            t_plot_min = self.t_min
        if t_plot_max is None:
            t_plot_max = self.t_max
        markers = ["^", "v"]
        colors = np.array(['red', 'blue', 'green','magenta','yellow','purple','cyan','teal'])
        t = np.arange(t_plot_min, t_plot_max)
        effective_mass = {}
        if model_type == None:
            raise TypeError(model_type,'you need to supply a correlator model in order to generate an eff mass plot for that correlator')
        elif model_type == 'simult_baryons':
            nucleon_corr_gv = self.nucleon_corr_gv
            xi_corr_gv      = self.xi_corr_gv
            lam_corr_gv     = self.lam_corr_gv
            sigma_corr_gv   = self.sigma_corr_gv
        elif model_type == 'gmo_ratio':
            nucleon_corr_gv = self.gmo_corr_gv

        elif model_type == 'xi':
            nucleon_corr_gv = None
            xi_corr_gv      = self.xi_corr_gv
            lam_corr_gv     = None
            sigma_corr_gv   = None
            effective_mass['xi'] = self.get_nucleon_effective_mass(xi_corr_gv)

        
        effective_mass['proton'] = self.get_nucleon_effective_mass(nucleon_corr_gv)
        effective_mass['xi'] = self.get_nucleon_effective_mass(xi_corr_gv)
        effective_mass['sigma'] = self.get_nucleon_effective_mass(sigma_corr_gv)
        effective_mass['lam'] = self.get_nucleon_effective_mass(lam_corr_gv)

        if effective_mass is None:
            return None

        y = {}
        y_err = {}
        lower_quantile = np.inf
        upper_quantile = -np.inf
        for i,baryon in enumerate(effective_mass.keys()):
            y[baryon] = {}
            y_err[baryon] = {}
            for j, key in enumerate(effective_mass[baryon].keys()):
                y[baryon][key] = gv.mean(effective_mass[baryon][key])[t]
                y_err[baryon][key] = gv.sdev(effective_mass[baryon][key])[t]
                lower_quantile = np.min([np.nanpercentile(y[baryon][key], 25), lower_quantile])
                upper_quantile = np.max([np.nanpercentile(y[baryon][key], 75), upper_quantile])

                plt.errorbar(x=t, y=y[baryon][key], xerr=0.0, yerr=y_err[baryon][key], fmt=markers[j%len(markers)], capsize=4.0,
                    color = colors[i%len(colors)], capthick=1.0, alpha=0.6, elinewidth=5.0, label=baryon+'_'+key)
            delta_quantile = upper_quantile - lower_quantile
            plt.ylim(lower_quantile - 0.5*delta_quantile,
                    upper_quantile + 0.5*delta_quantile)

        if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = self._generate_data_from_fit(model_type=model_type, t=t)

            for j, key in enumerate(fit_data_gv.keys()):
                eff_mass_fit = self.get_nucleon_effective_mass(fit_data_gv, dt)[key][1:-1]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(eff_mass_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(eff_mass_fit, 1), t[1:-1], pm(eff_mass_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(eff_mass_fit, -1), pm(eff_mass_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("Simultaneous fit to 4 baryons for $N_{states} = $%s" %(self.n_states['gmo']), fontsize = 24)

        plt.xlim(t_plot_min-0.5, t_plot_max-.5)
        # plt.ylim(0.55,0.9)
         # Get unique markers when making legend
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {}
        for j, handle in enumerate(handles):
            temp[labels[j]] = handle

        plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])
        # plt.legend()
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel('$M^{eff}_{baryon}$', fontsize = 24)
        fig = plt.gcf()
        # plt.savefig(fig_name)
        if show_plot == True: plt.show()
        else: plt.close()

        return fig

    def plot_stability(self, model_type=None, t_start=None, t_end=None, t_middle=None,
                       vary_start=True, show_plot=False, n_states_array=None,fig_name=None):


        # Set axes: first for quantity of interest (eg, E0)
        ax = plt.axes([0.10,0.20,0.7,0.7])

        # Markers for identifying n_states
        markers = ["^", ">", "v", "<"]

        # Color error bars by chi^2/dof
        cmap = matplotlib.cm.get_cmap('rainbow_r')
        norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)

        fit_data = {}
        if n_states_array is None:
            fit_data[self.n_states[model_type]] = None
        else:
            for n_state in n_states_array:
                fit_data[n_state] = None

        if n_states_array is None:
            spacing = [0]
        else:
            spacing = (np.arange(len(n_states_array)) - (len(n_states_array)-1)/2.0)/((len(n_states_array)-1)/2.0) *0.25

        # Make fits from [t, t_end], where t is >= t_end
        if vary_start:
            if t_start is None:
                t_start = self.t_min

            if t_end is None:
                t_end = self.t_range[model_type][1]

            if t_middle is None:
                t_middle = int((t_start + 2*t_end)/3)

            plt.title("Stability plot, varying start\n Fitting [%s, %s], $N_{states} =$ %s"
                      %("$t$", t_end, sorted(fit_data.keys())), fontsize = 24)

            t = np.arange(t_start, t_middle + 1)

        # Vary end point instead
        else:
            if t_start is None:
                t_start = self.t_range[model_type][0]

            if t_end is None:
                t_end = self.t_max

            if t_middle is None:
                t_middle = int((2*t_start + t_end)/3)

            plt.title("Stability plot, varying end\n Fitting [%s, %s], $N_{states} =$ %s"
                      %(t_start, "$t$", sorted(fit_data.keys())), fontsize = 24)
            t = np.arange(t_middle, t_end + 1)

        for key in list(fit_data.keys()):
            fit_data[key] = {
                'y' : np.array([]),
                'chi2/df' : np.array([]),
                'Q' : np.array([]),
                't' : np.array([])
            }

        for n_state in list(fit_data.keys()):
            n_states_dict = self.n_states.copy()
            n_states_dict[model_type] = n_state
            for ti in t:
                t_range = {key : self.t_range[key] for key in list(self.t_range.keys())}
                if vary_start:
                    t_range[model_type] = [ti, t_end]
                    temp_fit = self.get_fit(t_range, n_states_dict)
                else:
                    t_range[model_type] = [t_start, ti]
                    temp_fit = self.get_fit(t_range, n_states_dict)
                if temp_fit is not None:
                    if model_type == 'corr':
                        fit_data[n_state]['y'] = np.append(fit_data[n_state]['y'], temp_fit.p['E0'])
                    fit_data[n_state]['chi2/df'] = np.append(fit_data[n_state]['chi2/df'], temp_fit.chi2 / temp_fit.dof)
                    fit_data[n_state]['Q'] = np.append(fit_data[n_state]['Q'], temp_fit.Q)
                    fit_data[n_state]['t'] = np.append(fit_data[n_state]['t'], ti)


        # Color map for chi/df
        cmap = matplotlib.cm.get_cmap('rainbow')
        min_max = lambda x : [np.min(x), np.max(x)]
        #minimum, maximum = min_max(fit_data['chi2/df'])
        norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)

        for i, n_state in enumerate(sorted(fit_data.keys())):
            for j, ti in enumerate(fit_data[n_state]['t']):
                color = cmap(norm(fit_data[n_state]['chi2/df'][j]))
                y = gv.mean(fit_data[n_state]['y'][j])
                yerr = gv.sdev(fit_data[n_state]['y'][j])

                alpha = 0.05
                if vary_start and ti == self.t_range[model_type][0]:
                    alpha=0.35
                elif (not vary_start) and ti == self.t_range[model_type][1]:
                    alpha=0.35

                plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
                plt.axvline(ti+0.5, linestyle='--', alpha=alpha)

                ti = ti + spacing[i]
                plt.errorbar(ti, y, xerr = 0.0, yerr=yerr, fmt=markers[i%len(markers)], mec='k', mfc='white', ms=10.0,
                     capsize=5.0, capthick=1.0, elinewidth=5.0, alpha=0.9, ecolor=color, label=r"$N$=%s"%n_state)


        # Band for best result

        best_fit = self.get_fit()
        if model_type == 'corr':
            y_best = best_fit.p['E0']
            ylabel = r'$E_0$'

        tp = np.arange(t[0]-1, t[-1]+2)
        tlim = (tp[0], tp[-1])

        pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
        y2 = np.repeat(pm(y_best, 0), len(tp))
        y2_upper = np.repeat(pm(y_best, 1), len(tp))
        y2_lower = np.repeat(pm(y_best, -1), len(tp))

        # Ground state plot
        plt.plot(tp, y2, '--')
        plt.plot(tp, y2_upper, tp, y2_lower)
        plt.fill_between(tp, y2_lower, y2_upper, facecolor = 'yellow', alpha = 0.25)

        plt.ylabel(ylabel, fontsize=24)
        plt.xlim(tlim[0], tlim[-1])

        # Limit y-axis when comparing multiple states
        if n_states_array is not None:
            plt.ylim(pm(y_best, -5), pm(y_best, 5))

        # Get unique markers when making legend
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {}
        for j, handle in enumerate(handles):
            temp[labels[j]] = handle

        plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])

        ###
        # Set axes: next for Q-values
        axQ = plt.axes([0.10,0.10,0.7,0.10])

        for i, n_state in enumerate(sorted(fit_data.keys())):
            t = fit_data[n_state]['t']
            for ti in t:
                alpha = 0.05
                if vary_start and ti == self.t_range[model_type][0]:
                    alpha=0.35
                elif (not vary_start) and ti == self.t_range[model_type][1]:
                    alpha=0.35

                plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
                plt.axvline(ti+0.5, linestyle='--', alpha=alpha)


            t = t + spacing[i]
            y = gv.mean(fit_data[n_state]['Q'])
            yerr = gv.sdev(fit_data[n_state]['Q'])
            color_data = fit_data[n_state]['chi2/df']

            sc = plt.scatter(t, y, vmin=0.25, vmax=1.75, marker=markers[i%len(markers)], c=color_data, cmap=cmap)

        # Set labels etc
        plt.ylabel('$Q$', fontsize=24)
        plt.xlabel('$t$', fontsize=24)
        plt.ylim(-0.05, 1.05)
        plt.xlim(tlim[0], tlim[-1])

        ###
        # Set axes: colorbar
        axC = plt.axes([0.85,0.10,0.05,0.80])

        #t = fit_data['t']
        #y = gv.mean(fit_data['g_A'])
        #yerr = gv.sdev(fit_data['g_A'])
        #color_data = fit_data['chi2/df']
        #sc = plt.scatter(t, y, vmin=0.25, vmax=1.75, c=color_data, cmap=cmap)

        colorbar = matplotlib.colorbar.ColorbarBase(axC, cmap=cmap,
                                    norm=norm, orientation='vertical')
        colorbar.set_label(r'$\chi^2_\nu$', fontsize = 24)

        fig = plt.gcf()
        plt.savefig(fig_name)
        if show_plot == True: plt.show()
        else: plt.close()

        return fig


    def return_best_fit_info(self,bs=None):
        plt.axis('off')
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

        # place a text box in upper left in axes coords
        #plt.text(0.05, 0.05, str(fit_ensemble.get_fit(fit_ensemble.best_fit_time_range[0], fit_ensemble.best_fit_time_range[1])),
        #fontsize=14, horizontalalignment='left', verticalalignment='bottom', bbox=props)
        text = self.__str__(bs=bs).expandtabs()
        plt.text(0.0, 1.0, str(text),
                 fontsize=10, ha='left', va='top', family='monospace', bbox=props)

        plt.tight_layout()
        fig = plt.gcf()
        plt.close()

        return fig

    def make_plots(self,model_type=None, fig_name = None,show_all=False):
        plots = np.array([])
        plots = np.append(self.return_best_fit_info(), plots)

        # Create a plot of best and stability plots
        #plots = np.append(plots, self.plot_all_fits())
        plots = np.append(plots, self.plot_effective_mass(t_plot_min=0,t_plot_max=20,model_type=model_type,fig_name=fig_name,show_fit=True))

        return plots
    
    def make_gmo_plots(self,model_type=None, fig_name = None,show_all=False):
        plots = np.array([])
        plots = np.append(self.return_best_fit_info(), plots)

        # Create a plot of best and stability plots
        #plots = np.append(plots, self.plot_all_fits())
        gmo_eff_mass = self.get_gmo_effective(gmo_ratio=self.gmo_corr_gv)
        plots = np.append(plots,self.plot_delta_gmo(correlators_gv=self.gmo_corr_gv,t_plot_min=0,t_plot_max=20,model_type=model_type,fig_name = fig_name,show_fit=True))
        plots = np.append(plots,self.plot_gmo_effective_mass(effective_mass=gmo_eff_mass,t_plot_min=0,t_plot_max=20,model_type=model_type,show_fit=True,fig_name=fig_name))

        return plots

    def __str__(self,bs=None):
        output = "Model Type:" + str(self.model_type) 
        output = output+"\n"
        # output = "delta_GMO fit type:" + str(self.gmo_type) 
        output = output+"\n"
        if bs:
            output = output +"\t bs: True" 
        output = output+"\n"
        # if self.nucleon_corr_gv is not None:
        output = output + "\t N_{corr} = "+str(self.n_states[self.model_type])+"\t"
        output = output+"\n"
        # if self.nucleon_corr_gv is not None:
        output = output + "\t t_{corr} = "+str(self.t_range[self.model_type])
        # output += ma.GMO()
        output = output+"\n"
        output += "Fit results: \n"

        temp_fit = self.get_fit()
        if self.model_type != 'gmo_direct':
            output += "\t GMO rln = " 
            output= output+  str(temp_fit.p['lam_E0'] + 1/3* temp_fit.p['sigma_E0'] - 2/3*temp_fit.p['proton_E0'] - 2/3*temp_fit.p['xi_E0'])
            output = output+"\n"
            output += "\t M_4 posterior =    " 
            output += str(temp_fit.p['lam_E0'] - 3* temp_fit.p['sigma_E0'] + temp_fit.p['proton_E0'] + temp_fit.p['xi_E0'])
            output = output+"\n"
            output += "\t Centroid posterior =    " 
            output += str(1/8*temp_fit.p['lam_E0'] + 3/8* temp_fit.p['sigma_E0'] + 1/4*temp_fit.p['proton_E0'] + 1/4*temp_fit.p['xi_E0'])
            output = output+"\n"
            output+= str("\t lam_E0=  ")
            output+= str(temp_fit.p['lam_E0'])
            output = output+"\n" 
            output+= str("\t proton_E0=  ")
            output+= str(temp_fit.p['proton_E0'])
            output = output+"\n"
            output+= str("\t xi_E0=  ")
            output += str(temp_fit.p['xi_E0'])
            output = output+"\n"
            output+= str("\t sigma_E0=  ")
            output += str(temp_fit.p['sigma_E0'])
            output+= "\n"
        
        return output + str(temp_fit)
