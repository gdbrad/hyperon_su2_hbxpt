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

class InputOutput(object):


    def __init__(self):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

        with h5py.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        with h5py.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        ensembles.remove('a12m220')
        ensembles.remove('a12m220S')

        self.ensembles = ensembles
        self.project_path = project_path


    # Valid choices for scheme: 't0_org', 't0_imp', 'w0_org', 'w0_imp' (see hep-lat/2011.12166)
    def _get_bs_data(self, scheme=None):
        to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
        hbar_c = self.get_data_phys_point('hbarc') # MeV-fm (PDG 2019 conversion constant)

        if scheme is None:
            scheme = 'w0_imp'
        if scheme not in ['t0_org', 't0_imp', 'w0_org', 'w0_imp']:
            raise ValueError('Invalid scale setting scheme')

        data = {}
        with h5py.File(self.project_path+'/data/input_data.h5', 'r') as f: 
            for ens in self.ensembles:
                data[ens] = {}
                data[ens]['units_MeV'] = hbar_c / to_gvar(f[ens]['a_fm'][scheme][:])
                data[ens]['a/w'] = to_gvar(f[ens]['a_w'])
                data[ens]['alpha_s'] = f[ens]['alpha_s']
                data[ens]['L'] = f[ens]['L']
                data[ens]['m_pi'] = f[ens]['mpi'][:]
                data[ens]['m_k'] = f[ens]['mk'][:]
                data[ens]['lam_chi'] = 4 *np.pi *f[ens]['Fpi'][:]

        with h5py.File(self.project_path+'/data/hyperon_data.h5', 'r') as f:
            for ens in self.ensembles:
                if ens+'_hp' in list(f):
                    for obs in list(f[ens+'_hp']):
                        data[ens][obs] = f[ens+'_hp'][obs][:]
                else:
                    for obs in list(f[ens]):
                        data[ens][obs] = f[ens][obs][:]


        return data
        

    def get_data(self, scheme=None):
        bs_data = self._get_bs_data(scheme)

        gv_data = {}
        dim1_obs = ['m_delta', 'm_lam', 'm_sigma', 'm_sigma_st', 'm_xi', 'm_xi_st', 'm_pi', 'm_k', 'lam_chi']
        for ens in self.ensembles:
            gv_data[ens] = {}
            for obs in dim1_obs:
                gv_data[ens][obs] = bs_data[ens][obs] - np.mean(bs_data[ens][obs]) + bs_data[ens][obs][0]

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 
            for obs in dim1_obs:
                gv_data[ens][obs] = gv_data[ens][obs] *bs_data[ens]['units_MeV']

            gv_data[ens]['a/w'] = bs_data[ens]['a/w']

        ensembles = list(gv_data)
        output = {}
        for param in gv_data[self.ensembles[0]]:
            output[param] = np.array([gv_data[ens][param] for ens in self.ensembles])
        return output, ensembles


    def get_data_phys_point(self, param=None):
        data_phys_point = {
            'a/w' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269804, 0), # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'

            'm_xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'm_xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
            #'mss' : gv.gvar('688.5(2.2)'), # Taken from arxiv/1303.1670
        }
        if param is not None:
            return data_phys_point[param]
        return data_phys_point

    
            