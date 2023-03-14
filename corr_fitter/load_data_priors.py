import pandas as pd
import gvar as gv 
import h5py as h5 
import numpy as np 
import os 
import bs_utils as bs 


# def make_prior_from_fit(fit):

#         output = {}
#         energy_levels = {
#         'lam': gv.gvar(1115.683, 0.006),
#         'sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
#         'sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
#         'xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
#         'xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
#         'proton' : gv.gvar(938.272,.0000058)
#         }

#         fit_parameters = fit.p
#         for key in fit_parameters:
#             for corr in ['delta', 'lam', 'proton', 'sigma', 'sigma_st', 'xi_st', 'xi']:
#                 if key in [corr+'_E0',corr+'_log(dE)']: 
#                     output[corr+'_E'] =  energy_levels[corr]*gv.mean(fit_parameters['E0']),
#                     np.repeat(gv.mean(fit_parameters[corr+'E0']) * 350.0/ 938.0, 4))



#                 # if key == 'log(E0)' or key == 'E0':
#                 'm_lambda' : ,
                

#                 # # Only works for protons
#                 # # In order: proton, Roper resonance, two pions, L=1 pion excitation

#                 #     rough_energy_levels = np.array([938.0, 1440, 938+2*350,  938+2*350+110]) / 938.0
#                 #     output['E'] = gv.gvar(rough_energy_levels*gv.mean(fit_parameters['E0']),
#                 #                             np.repeat(gv.mean(fit_parameters['E0']) * 350.0/ 938.0, 4))

#                 if key in [corr+'_z_PS']:
#                     wf_dir = gv.gvar(0, 2*gv.mean(fit_parameters[corr+'_z_PS'][0]))
#                     output[corr+'_z_PS'] = np.repeat(wf_dir, 4)

#                 if key in  [corr+'_z_PS']:
#                     wf_smr = gv.gvar(gv.mean(fit_parameters[corr+'_z_SS'][0]), gv.mean(fit_parameters[corr+'_z_SS'][0]))
#                     output[corr+'_z_SS'] = np.repeat(wf_smr, 4)

#     #         elif key == 'd_A_dir':
#     #             d_n = gv.mean(self.axial_fh_num_gv['dir'][1] * np.exp(fit_parameters['E0']))
#     #             output['d_A_dir'] = np.repeat(gv.gvar(d_n, d_n), 4)

#     #         elif key == 'd_A_smr':
#     #             d_n = gv.mean(self.axial_fh_num_gv['smr'][1] * np.exp(fit_parameters['E0']))
#     #             output['d_A_smr'] = np.repeat(gv.gvar(d_n, d_n), 4)
#     # s        elif key == 'd_V_dir':
#     #             d_n = gv.mean(self.vector_fh_num_gv['dir'][1] * np.exp(fit_parameters['E0']))
#     #             output['d_V_dir'] = np.repeat(gv.gvar(d_n, d_n), 4)

#     #         elif key == 'd_V_smr':
#     #             d_n = gv.mean(self.vector_fh_num_gv['smr'][1] * np.exp(fit_parameters['E0']))
#     #             output['d_V_smr'] = np.repeat(gv.gvar(d_n, d_n), 4)

#     #         elif key == 'g_A_nm':
#     #             gA = gv.mean(fit_parameters['g_A_nm'][0, 0])
#     #             temp_array = gv.gvar(np.repeat(0, 16), np.repeat(1, 16))
#     #             temp_array[0] = gv.gvar(gA, 0.40*gA)
#     #             temp_array = np.reshape(temp_array, (4, 4))
#     #             output['g_A_nm'] = temp_array

#     #         elif key == 'g_V_nm':
#     #             gV = gv.mean(fit_parameters['g_V_nm'][0, 0])
#     #             temp_array = gv.gvar(np.repeat(0, 16), np.repeat(1, 16))
#     #             temp_array[0] = gv.gvar(gV, 0.40*gV)
#     #             temp_array = np.reshape(temp_array, (4, 4))
#     #             output['g_V_nm'] = temp_array

#         return output

def pickle_out(fit_out,out_path,species=None):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fit_dump = {}
    fit_dump['prior'] = fit_out.prior
    fit_dump['p'] = fit_out.p
    fit_dump['logGBF'] = fit_out.logGBF
    fit_dump['Q'] = fit_out.Q
    if species == 'meson':
        return gv.dump(fit_dump,out_path+'meson_fit_params')
    elif species == 'baryon':
        return gv.dump(fit_dump,out_path+'fit_params')
    elif species == 'baryon_w_gmo':
        return gv.dump(fit_dump,out_path+'fit_params_all')

def print_posterior(out_path):
    posterior = {}
    post_out = gv.load(out_path+"fit_params")
    posterior['lam_E0'] = post_out['p']['lam_E0']
    posterior['lam_E1'] = np.exp(post_out['p']['lam_log(dE)'][0])+posterior['lam_E0']
    posterior['proton_E0'] = post_out['p']['proton_E0']
    posterior['proton_E1'] = np.exp(post_out['p']['proton_log(dE)'][0])+posterior['proton_E0']
    posterior['sigma_E0'] = post_out['p']['sigma_E0']
    posterior['sigma_E1'] = np.exp(post_out['p']['sigma_log(dE)'][0]) + posterior['sigma_E0']
    posterior['xi_E0'] = post_out['p']['xi_E0']
    posterior['xi_E1'] = np.exp(post_out['p']['xi_log(dE)'][0]) + posterior['xi_E0']

    return posterior
def get_raw_corr(file_h5,abbr,particle):
    data = {}
    particle_path = '/'+abbr+'/'+particle
    with h5.File(file_h5,"r") as f:
        if f[particle_path].shape[3] == 1:
            data['SS'] = f[particle_path][:, :, 0, 0].real
            data['PS'] = f[particle_path][:, :, 1, 0].real 
    return data

def get_raw_corr_new(file_h5,abbr):
    data = {}
    with h5.File(file_h5,"r") as f:
        for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
            particle_path = '/'+abbr+'/'+baryon
            data[baryon+'_SS'] = f[particle_path][:, :, 0, 0].real
            data[baryon+'_PS'] = f[particle_path][:, :, 1, 0].real 
    return data

def resample_correlator(raw_corr,bs_list, n):
    resampled_raw_corr_data = ({key : raw_corr[key][bs_list[n, :], :]
    for key in raw_corr.keys()})
    resampled_corr_gv = resampled_raw_corr_data
    return resampled_corr_gv

def fetch_prior(model_type,p_dict):

    prior_nucl = {}
    prior = {}
    # prior_xi = {}
    states= p_dict[str(model_type)]
    newlist = [x for x in states]
    for x in newlist:
        path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
        df = pd.read_csv(path, index_col=0).to_dict()
        for key in list(df.keys()):
            length = int(np.sqrt(len(list(df[key].values()))))
            prior_nucl[key] = list(df[key].values())[:length]
        prior = gv.gvar(prior_nucl)
    return prior

# def get_data_phys_pt()
