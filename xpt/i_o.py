import numpy as np
import gvar as gv
import sys
import datetime
import re
import os
#import yaml
import h5py

# Set defaults for plots
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = (6.75, 6.75/1.618034333)
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

# TODO:
# add results summary 
# add model collection based on particle names and model name

class InputOutput(object):


    def __init__(self):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

        # add in defualt dict

        with h5py.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        with h5py.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        ensembles.remove('a12m220')
        ensembles.remove('a12m220ms')
        ensembles.remove('a12m310XL')
        ensembles.remove('a12m220S')
        ensembles.remove('a12m180L')

        self.ensembles = ensembles
        self.project_path = project_path

    # Valid choices for scheme: 't0_org', 't0_imp', 'w0_org', 'w0_imp' (see hep-lat/2011.12166)
    def _get_bs_data(self, scheme=None, units=None):
        to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
        hbar_c = self.get_data_phys_point('hbarc') # MeV-fm (PDG 2019 conversion constant)
        scale_factors = gv.load(self.project_path +'/data/scale_setting.p')

        if scheme is None:
            scheme = 'w0_imp'
        if scheme not in ['t0_org', 't0_imp', 'w0_org', 'w0_imp']:
            raise ValueError('Invalid scale setting scheme')

        data = {}
        with h5py.File(self.project_path+'/data/input_data.h5', 'r') as f: 
            for ens in self.ensembles:
                data[ens] = {}
                if scheme in ['w0_org','w0_imp'] and units=='phys':
                    data[ens]['units'] = hbar_c *scale_factors[scheme+':'+ens[:3]] /scale_factors[scheme+':w0']
                elif scheme in ['w0_org', 'w0_imp'] and units=='w0':
                    data[ens]['units'] = hbar_c *scale_factors[scheme+':'+ens[:3]]
                elif scheme in ['t0_org', 't0_imp'] and units=='phys':
                    data[ens]['units'] = hbar_c / to_gvar(f[ens]['a_fm'][scheme][:]) 
                data[ens]['units_MeV'] = hbar_c / to_gvar(f[ens]['a_fm'][scheme][:])
                data[ens]['alpha_s'] = f[ens]['alpha_s']
                data[ens]['L'] = f[ens]['L']
                data[ens]['m_pi'] = f[ens]['mpi'][:]
                data[ens]['m_k'] = f[ens]['mk'][:]
                data[ens]['lam_chi'] = 4 *np.pi *f[ens]['Fpi'][:]

                if units=='Fpi':
                    #for data[ens]['units'] not in data[ens]['lam_chi']:
                    data[ens]['units'] = 1/data[ens]['lam_chi'] #for removing lam_chi dependence of fits

                if scheme == 'w0_imp':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat_imp']))**2
                elif scheme ==  'w0_org':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat']))**2
                elif scheme == 't0_imp':
                    data[ens]['eps2_a'] = 1 / (4 *to_gvar(f[ens]['t0aSq_imp']))
                elif scheme == 't0_org':
                    data[ens]['eps2_a'] = 1 / (4 *to_gvar(f[ens]['t0aSq']))

        with h5py.File(self.project_path+'/data/hyperon_data.h5', 'r') as f:
            for ens in self.ensembles:
                for obs in list(f[ens]):
                    data[ens].update({obs : f[ens][obs][:]})
                if ens+'_hp' in list(f):
                    for obs in list(f[ens+'_hp']):
                        data[ens].update({obs : f[ens+'_hp'][obs][:]})

        return data


    def get_data(self, scheme=None,units=None):
        bs_data = self._get_bs_data(scheme,units)
        phys_data = self.get_data_phys_point(param='m_proton')

        gv_data = {}
        dim1_obs = ['m_proton', 'm_delta', 'm_lambda', 'm_sigma', 'm_sigma_st', 'm_xi', 'm_xi_st', 'm_omega', 'm_pi', 'm_k', 'lam_chi']
        for ens in self.ensembles:
            gv_data[ens] = {}
            for obs in dim1_obs:
                gv_data[ens][obs] = bs_data[ens][obs] - np.mean(bs_data[ens][obs]) + bs_data[ens][obs][0]

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 
            for obs in dim1_obs:
                gv_data[ens][obs] = gv_data[ens][obs] *bs_data[ens]['units_MeV']

            gv_data[ens]['eps2_a'] = bs_data[ens]['eps2_a']
            #gv_data[ens]['m_proton_phys'] = phys_data[ens]


        ensembles = list(gv_data)
        output = {}
        for param in gv_data[self.ensembles[0]]:
            output[param] = np.array([gv_data[ens][param] for ens in self.ensembles])
        return output, ensembles


    def get_data_phys_point(self, param=None):
        data_phys_point = {
            'eps2_a' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269804, 0), # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'

            'm_lambda' : gv.gvar(1115.683, 0.006),
            'm_sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
            'm_sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
            'm_xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'm_xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
            'm_omega' : gv.gvar(1672.45,29),
            'm_proton' : gv.gvar(938.272,.0000058)
        }
        if param is not None:
            return data_phys_point[param]
        return data_phys_point

    def get_posterior(self,fit_test=None,prior=None,param=None):
        if param == 'all':
            return fit_test.fit.p
        elif param is not None:
            return fit_test.fit.p[param]
        else:
            output = {}
            for param in prior:
                if param in fit_test.fit.p:
                    output[param] = fit_test.fit.p[param]
            return output

    def make_prior(self,data,prior):
        new_prior = {}
        for key in prior:
            new_prior[key] = prior[key]
        for key in ['m_pi', 'm_k', 'lam_chi', 'eps2_a','m_delta']:
            new_prior[key] = data[key]
        return new_prior




