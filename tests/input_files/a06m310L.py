import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a06m310L',
    'hyperons' : ['lambda_z', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma' : [12,20],
        'proton' : [12,20],
        'delta' : [12,20],
        'xi' :  [12,20],
        'xi_st' : [12,20],
        'sigma_st' : [12,20],
        'lam' : [12,20],
        'pi' : [5,30],
        'kplus': [8,28],
        'hyperons':   [12,20],
        'all':   [12,20]
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
    'n_states' : {
        'sigma' : 2,
        'delta': 2,
        'proton': 2,
        'sigma_st':2,
        'xi' :2,
        'xi_st':2,
        'lam':2,
        'pi' : 2,
        'kplus': 2,
        'hyperons': 2,
        'all':2

    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

prior = gv.BufferDict()
prior = {
    'proton_E': np.array(['0.3(30)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.35(3.3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'delta_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.4(30)', '0.5(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.45(3)', '0.6(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_E': np.array(['0.4(3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.4(3)', '0.6(32)', '0.8(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.4(3)', '0.7(32)', '0.8(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    }
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


# TODO put prior routines in here, filename save options 
