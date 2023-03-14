import lsqfit
import numpy as np 
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt

class fitter(object):

    def __init__(self, n_states,prior, t_period,t_range,states,
                 p_dict=None,nucleon_corr=None,lam_corr=None,
                 xi_corr=None,xi_st_corr=None,sigma_corr=None,
                 sigma_st_corr=None,delta_corr=None,model_type=None,simult=None):

        self.n_states = n_states
        self.t_period = t_period
        self.t_range = t_range
        self.prior = prior
        self.p_dict = p_dict
        self.lam_corr=lam_corr
        self.sigma_corr=sigma_corr
        self.nucleon_corr=nucleon_corr
        self.xi_corr=xi_corr
        self.xi_st_corr = xi_st_corr
        self.sigma_st_corr = sigma_st_corr
        self.delta_corr = delta_corr
        self.fit = None
        self.model_type = model_type
        self.simult = simult
        self.states = states
        self.prior = self._make_prior(prior)
        effective_mass = {}
        self.effective_mass = effective_mass
        # self.fits = {}
        # self.extrapolate = None
        # self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}

    def return_best_fit_info(self):
        plt.axis('off')
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        #plt.text(0.05, 0.05, str(fit_ensemble.get_fit(fit_ensemble.best_fit_time_range[0], fit_ensemble.best_fit_time_range[1])),
        #fontsize=14, horizontalalignment='left', verticalalignment='bottom', bbox=props)
        text = self.__str__().expandtabs()
        plt.text(0.0, 1.0, str(text),
                 fontsize=16, ha='left', va='top', family='monospace', bbox=props)

        plt.tight_layout()
        fig = plt.gcf()
        plt.close()

        return fig

    def __str__(self):
        output = "Model Type:" + str(self.model_type) 
        output = output+"\n"

        output = output + "\t N_{corr} = "+str(self.n_states[self.model_type])+"\t"
        output = output+"\n"
        output += "Fit results: \n"

        output += str(self.get_fit())
        return output


    def _generate_data_from_fit(self, t,fit=None, t_start=None, t_end=None, model_type=None, n_states=None):
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

        # models = self._get_models(model_type=model_type)
        if fit is None:

            fit = self.get_fit(t_range=t_range, n_states=n_states)

        # datatag[-3:] converts, eg, 'nucleon_dir' -> 'dir'
        output = {model.datatag : model.fitfcn(p=fit.p, t=t) for model in models}
        return output


    def get_fit(self):
        if self.fit is not None:
            return self.fit
        else:
            return self._make_fit()

    def fcn_effective_mass(self, t, t_start=None, t_end=None, n_states=None):
        if t_start is None:
            t_start = self.t_range[self.model_type][0]
        if t_end is None:
            t_end = self.t_range[self.model_type][1]
        if n_states is None: n_states = self.n_states

        p = self.get_fit().p
        output = {}
        for model in self._make_models_simult_fit():
            snk = model.datatag
            output[snk] = model.fcn_effective_mass(p, t)
        return output

    def plot_effective_mass(self, tmin=None, tmax=None, ylim=None, show_fit=True,show_plot=True,fig_name=None):
        if tmin is None: tmin = 1
        if tmax is None: tmax = self.t_period - 1

        fig = self._plot_quantity(
            quantity=self.effective_mass, 
            ylabel=r'$m_\mathrm{eff}$', 
            tmin=tmin, tmax=tmax, ylim=ylim) 

        if show_fit:
            ax = plt.gca()

            colors = ['rebeccapurple', 'mediumseagreen']
            t = np.linspace(tmin, tmax)
            effective_mass_fit = self.fcn_effective_mass(t=t)
            for j, snk in enumerate(sorted(effective_mass_fit)):
                color = colors[j%len(colors)]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                ax.plot(t, pm(effective_mass_fit[snk], 0), '--', color=color)
                ax.plot(t, pm(effective_mass_fit[snk], 1), 
                            t, pm(effective_mass_fit[snk], -1), color=color)
                ax.fill_between(t, pm(effective_mass_fit[snk], -1), pm(effective_mass_fit[snk], 1), facecolor=color, alpha = 0.10, rasterized=True)

        fig = plt.gcf()
        if show_plot:
            plt.show()
        plt.close()
        return fig

    def _plot_quantity(self, quantity,
            tmin, tmax, ylabel=None, ylim=None):

        fig, ax = plt.subplots()
        
        colors = ['rebeccapurple', 'mediumseagreen']
        for j, snk in enumerate(sorted(quantity)):
            x = np.arange(tmin, tmax)
            y = gv.mean(quantity[snk])[x]
            y_err = gv.sdev(quantity[snk])[x]

            ax.errorbar(x, y, xerr = 0.0, yerr=y_err, fmt='o', capsize=5.0,
                        color=colors[j%len(quantity)], capthick=2.0, alpha=0.6, elinewidth=5.0, label=snk)

        # Label dirac/smeared data
        #plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.legend(loc=3, bbox_to_anchor=(0,1), ncol=len(quantity))
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)

        if ylim is not None:
            plt.ylim(ylim)
        fig = plt.gcf()
        return fig

    def get_energies(self):

        # Don't rerun the fit if it's already been made
        if self.fit is not None:
            temp_fit = self.fit
        else:
            temp_fit = self.get_fit()

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        output = gv.gvar(np.zeros(max_n_states))
        output[0] = temp_fit.p['E0']
        for k in range(1, max_n_states):
            output[k] = output[0] + np.sum([(temp_fit.p['dE'][j]) for j in range(k)], axis=0)
        return output

    def _make_fit(self):
        # Essentially: first we create a model (which is a subclass of MultiFitter)
        # Then we make a fitter using the models
        # Finally, we make the fit with our two sets of correlators

        models = self._make_models_simult_fit()
        data = self._make_data()
        fitter_ = lsqfit.MultiFitter(models=models)
        fit = fitter_.lsqfit(data=data, prior=self.prior)
        self.fit = fit
        return fit

    def _make_models_simult_fit(self):
        models = np.array([])

        if self.nucleon_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.nucleon_corr.keys()):
                param_keys = {
                    'E0'      : 'proton_E0',
                    'log(dE)' : 'proton_log(dE)',
                    'z'       : 'proton_z_'+sink 
                }   
                models = np.append(models,
                        baryon_model(datatag="nucleon_"+sink,
                        t=list(range(self.t_range['proton'][0], self.t_range['proton'][1])),
                        param_keys=param_keys, n_states=self.n_states['proton']))
        
        if self.delta_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.delta_corr.keys()):
                param_keys = {
                    'E0'      : 'delta_E0',
                    'log(dE)' : 'delta_log(dE)',
                    'z'       : 'delta_z_'+sink 
                }   
                models = np.append(models,
                        baryon_model(datatag="delta_"+sink,
                        t=list(range(self.t_range['delta'][0], self.t_range['delta'][1])),
                        param_keys=param_keys, n_states=self.n_states['delta']))
        
        if self.lam_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.lam_corr.keys()):
                param_keys = {
                    'E0'      : 'lam_E0',
                    'log(dE)' : 'lam_log(dE)',
                    'z'      : 'lam_z_'+sink 
                }   
                models = np.append(models,
                        baryon_model(datatag="lam_"+sink,
                        t=list(range(self.t_range['lam'][0], self.t_range['lam'][1])),
                        param_keys=param_keys, n_states=self.n_states['lam']))
        if self.xi_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.xi_corr.keys()):
                param_keys = {
                    'E0'      : 'xi_E0',
                    'log(dE)' : 'xi_log(dE)',
                    'z'      :  'xi_z_'+sink 
                }   
                models = np.append(models,
                        baryon_model(datatag="xi_"+sink,
                        t=list(range(self.t_range['xi'][0], self.t_range['xi'][1])),
                        param_keys=param_keys, n_states=self.n_states['xi']))
        if self.xi_st_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.xi_st_corr.keys()):
                param_keys = {
                    'E0'      : 'xi_st_E0',
                    'log(dE)' : 'xi_st_log(dE)',
                    'z'      : 'xi_st_z_'+sink
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        baryon_model(datatag="xi_st_"+sink,
                        t=list(range(self.t_range['xi_st'][0], self.t_range['xi_st'][1])),
                        param_keys=param_keys, n_states=self.n_states['xi_st']))
        if self.sigma_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.sigma_corr.keys()):
                param_keys = {
                    'E0'      : 'sigma_E0',
                    'log(dE)' : 'sigma_log(dE)',
                    'z'       : 'sigma_z_'+sink 
                }   
                models = np.append(models,
                        baryon_model(datatag="sigma_"+sink,
                        t=list(range(self.t_range['sigma_st'][0], self.t_range['sigma_st'][1])),
                        param_keys=param_keys, n_states=self.n_states['sigma_st']))
        if self.sigma_st_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.sigma_corr.keys()):
                param_keys = {
                    'E0'      : 'sigma_st_E0',
                    'log(dE)' : 'sigma_st_log(dE)',
                    'z'       : 'sigma_st_z_'+sink 
                }   
                models = np.append(models,
                        baryon_model(datatag="sigma_st_"+sink,
                        t=list(range(self.t_range['sigma_st'][0], self.t_range['sigma_st'][1])),
                        param_keys=param_keys, n_states=self.n_states['sigma_st']))
        
        return models

    # data array needs to match size of t array
    def _make_data(self):
        data = {}
        if self.nucleon_corr is not None:
            for sinksrc in list(self.nucleon_corr.keys()):
                data["nucleon_"+sinksrc] = self.nucleon_corr[sinksrc][self.t_range['proton'][0]:self.t_range['proton'][1]]
        if self.delta_corr is not None:
            for sinksrc in list(self.delta_corr.keys()):
                data["delta_"+sinksrc] = self.delta_corr[sinksrc][self.t_range['delta'][0]:self.t_range['delta'][1]]
        if self.lam_corr is not None:
            for sinksrc in list(self.lam_corr.keys()):
                data["lam_"+sinksrc] = self.lam_corr[sinksrc][self.t_range['lam'][0]:self.t_range['lam'][1]]
        if self.sigma_corr is not None:
            for sinksrc in list(self.sigma_corr.keys()):
                data["sigma_"+sinksrc] = self.sigma_corr[sinksrc][self.t_range['sigma'][0]:self.t_range['sigma'][1]]
        if self.sigma_st_corr is not None:
            for sinksrc in list(self.sigma_st_corr.keys()):
                data["sigma_st_"+sinksrc] = self.sigma_st_corr[sinksrc][self.t_range['sigma_st'][0]:self.t_range['sigma_st'][1]]
        if self.xi_corr is not None:
            for sinksrc in list(self.xi_corr.keys()):
                data["xi_"+sinksrc] = self.xi_corr[sinksrc][self.t_range['xi'][0]:self.t_range['xi'][1]]
        if self.xi_st_corr is not None:
            for sinksrc in list(self.xi_st_corr.keys()):
                data["xi_st_"+sinksrc] = self.xi_st_corr[sinksrc][self.t_range['xi_st'][0]:self.t_range['xi_st'][1]]
        return data
    
    def _make_prior(self,prior):
        resized_prior = {}

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        for key in list(prior.keys()):
            resized_prior[key] = prior[key][:max_n_states]

        new_prior = resized_prior.copy()
        if self.simult:
            for corr in ['sigma','lam','proton','xi','xi_st','sigma_st','delta']:
                new_prior[corr+'_E0'] = resized_prior[corr+'_E'][0]
                new_prior.pop(corr+'_E', None)
                new_prior[corr+'_log(dE)'] = gv.gvar(np.zeros(len(resized_prior[corr+'_E']) - 1))
                for j in range(len(new_prior[corr+'_log(dE)'])):
                    temp = gv.gvar(resized_prior[corr+'_E'][j+1]) - gv.gvar(resized_prior[corr+'_E'][j])
                    temp2 = gv.gvar(resized_prior[corr+'_E'][j+1])
                    temp_gvar = gv.gvar(temp.mean,temp2.sdev)
                    new_prior[corr+'_log(dE)'][j] = np.log(temp_gvar)
        else:
            for corr in self.states:
                new_prior[corr+'_E0'] = resized_prior[corr+'_E'][0]
                new_prior.pop(corr+'_E', None)

        # We force the energy to be positive by using the log-normal dist of dE
        # let log(dE) ~ eta; then dE ~ e^eta
                new_prior[corr+'_log(dE)'] = gv.gvar(np.zeros(len(resized_prior[corr+'_E']) - 1))
                for j in range(len(new_prior[corr+'_log(dE)'])):
                    temp = gv.gvar(resized_prior[corr+'_E'][j+1]) - gv.gvar(resized_prior[corr+'_E'][j])
                    temp2 = gv.gvar(resized_prior[corr+'_E'][j+1])
                    temp_gvar = gv.gvar(temp.mean,temp2.sdev)
                    new_prior[corr+'_log(dE)'][j] = np.log(temp_gvar)

        return new_prior

class baryon_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(baryon_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t
        z = p[self.param_keys['z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['E0']]
        log_dE = p[self.param_keys['log(dE)']]
        output = z[0] * np.exp(-E0 * t)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        t = np.array(t)
        
        return np.exp(self.fcn_effective_mass(p, t)*t) * self.fitfcn(p, t)

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]


class proton_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(proton_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t
        z = p[self.param_keys['proton_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['proton_E0']]
        log_dE = p[self.param_keys['proton_log(dE)']]
        output = z[0] * np.exp(-E0 * t)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        t = np.array(t)
        
        return np.exp(self.fcn_effective_mass(p, t)*t) * self.fitfcn(p, t)

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

class lam_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(lam_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        # z_PS = p[self.param_keys['z_PS']]
        # z_SS = p[self.param_keys['z_SS']]
        z = p[self.param_keys['lam_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['lam_E0']]
        log_dE = p[self.param_keys['lam_log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

class xi_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(xi_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        # z_PS = p[self.param_keys['z_PS']]
        # z_SS = p[self.param_keys['z_SS']]
        z = p[self.param_keys['xi_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['xi_E0']]
        log_dE = p[self.param_keys['xi_log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

    

class sigma_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(sigma_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        # z_PS = p[self.param_keys['z_PS']]
        # z_SS = p[self.param_keys['z_SS']]
        z = p[self.param_keys['sigma_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['sigma_E0']]
        log_dE = p[self.param_keys['sigma_log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]