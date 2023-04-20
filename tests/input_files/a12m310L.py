import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a12m310L',
    'part' : ['delta_pp', 'kplus', 'lambda_z', 'omega_m', 'piplus', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'particles' : ['proton'],
    'meson_states' : ['piplus','kplus'],
    'hyperons' : ['delta_pp', 'lambda_z', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma' : [5,18],
        'xi' :  [5,18],
        'xi_st' : [5,18],
        'sigma_st' : [5,18],
        'proton' :   [5,18],
        'delta' : [5,18],
        'lam' : [5,18],
        'pi' : [5,30],
        'kplus': [8,28],
        'hyperons':   [5,18],
        'all':   [5,18]
    },
    'n_states' : {
        'sigma' : 2,
        'sigma_st':2,
        'xi' :2,
        'xi_st':2,
        'delta':2,
        'proton':2,
        'lam':2,
        'pi' : 2,
        'kplus': 2,
        'hyperons': 2,
        'all':2

    },
     'tag':{
        'sigma' : 'sigma',
        'sigma_st' : 'sigma_st',
        'xi' :  'xi',
        'xi_st' : 'xi_st',
        'lam' : 'lam',
        'proton': 'proton',
        'delta' : 'delta'
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
    }

prior = gv.BufferDict()
prior = {
    'sigma_E': np.array(['0.73(22)', '0.8(3.2)', '0.9(3.2)', '1.0(3.2)'], dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'sigma_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-07', '4.4(4.4)e-07'],dtype=object),
    'sigma_st_E': np.array(['0.8(30)', '0.9(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'lam_E': np.array(['0.7(2.2)', '0.9(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'proton_E': np.array(['0.61(22)', '0.9(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07'],dtype=object),
    'xi_E': np.array(['0.8(2.2)', '1.28(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'xi_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07'],dtype=object),
    'xi_st_E': np.array(['0.8(22)', '0.9(32)', '1.0(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'delta_E': np.array(['0.91(22)', '0.95(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'delta_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object)}
'''
$\delta_{GMO}$ xpt extrapolation model and prior information
'''
model_info = {}
model_info['particles'] = ['piplus','kplus','eta']
model_info['order_chiral'] = 'lo'
model_info['tree_level'] = True
model_info['loop_level'] = False
model_info['delta'] = True
model_info['abbr'] = ['a12m180L']
model_info['observable'] = ['delta_gmo'] #'centroid', 'octet'

tag = {
    'sigma': 'sigma',
    'sigma_st': 'sigma_st',
    'xi': 'xi',
    'xi_st': 'xi_st',
    'lam': 'lam',
}
