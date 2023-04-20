import lsqfit
import numpy as np 
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt

class fitter(object):

    def __init__(self, n_states,prior, t_period,t_range,states,
                 p_dict=None,raw_corrs=None,model_type=None,simult=None):

        self.n_states = n_states
        self.t_period = t_period
        self.t_range = t_range
        self.prior = prior
        self.p_dict = p_dict
        self.raw_corrs = raw_corrs
        self.fit = None
        self.model_type = model_type
        self.simult = simult
        self.states = states
        self.prior = self._make_prior(prior)
        effective_mass = {}
        self.effective_mass = effective_mass
    
    def __str__(self):
        output = "Model Type:" + str(self.model_type) 
        output = output+"\n"

        output = output + "\t N_{corr} = "+str(self.n_states[self.model_type])+"\t"
        output = output+"\n"
        output += "Fit results: \n"

        output += str(self.get_fit())
        return output

    def get_fit(self):
        if self.fit is not None:
            return self.fit
        else:
            return self._make_fit()

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

        if self.raw_corrs is not None:
                for corr in self.raw_corrs:
                    for sink in list(['SS','PS']):
                        datatag = self.p_dict['tag'][corr]
                        param_keys = {
                            'E0'      : datatag+'_E0',
                            'log(dE)' : datatag+'_log(dE)',
                            'z'       : datatag+'_z_'+sink
                        }
                        models = np.append(models,
                                baryon_model(datatag=datatag+"_"+sink,
                                t=list(range(self.t_range[datatag][0], self.t_range[datatag][1])),
                                param_keys=param_keys, n_states=self.n_states[datatag]))
        return models 

    # data array needs to match size of t array
    def _make_data(self):
        data = {}
        for corr_type in ['lam', 'sigma', 'sigma_st', 'xi', 'xi_st','proton','delta']:
            for sinksrc in list(['SS','PS']):
                data[corr_type + '_' + sinksrc] = self.raw_corrs[corr_type][sinksrc][self.t_range[corr_type][0]:self.t_range[corr_type][1]]
        return data

    def _make_prior(self,prior):
        resized_prior = {}

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        for key in list(prior.keys()):
            resized_prior[key] = prior[key][:max_n_states]

        new_prior = resized_prior.copy()
        if self.simult:
            for corr in ['sigma','lam','xi','xi_st','sigma_st','proton','delta']:
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