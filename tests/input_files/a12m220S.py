import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a12m220S', #CHANGE THIS
    'part' : ['delta_pp', 'kplus', 'lambda_z', 'omega_m', 'piplus', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'particles' : ['proton'],
    'meson_states' : ['piplus','kplus'],
    'gmo_direct': ['gmo'],
    'simult_baryons': ['sigma_p','lambda_z','proton','xi_z'],
    'simult_baryons_gmo': ['sigma_p','lambda_z','proton','xi_z'], #states for gmo study
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma' : [5,20],
        'xi' : [5,20],
        'proton' : [5,20],
        'delta' : [6,15],
        'lam' : [5,20],
        'gmo' : [2,10], 
        'pi' : [5,30],
        'kplus': [8,28],
	    'gmo_ratio':[5,20],
        'gmo_direct':[5,20],
        'simult_baryons': [4,15],
        'simult_baryons_gmo':[4,15]
    },
    'n_states' : {
        'sigma' : 2,
        'xi' :2,
        'delta':2,
        'proton':2,
        'lam':2,
        'gmo':2,
        'pi' : 2,
        'kplus': 2,
	    'gmo_ratio':2,
	    'gmo_direct':2,
        'simult_baryons':2,
        'simult_baryons_gmo':2
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

prior = gv.BufferDict()
prior = {
    'gmo_E': np.array(['0.001(15)', '0.002(15)'], dtype=object),
    'gmo_z_PS': np.array(['0.7(7.0)', '0.7(7.2)'], dtype=object),
    'gmo_z_SS': np.array(['0.7(7.0)', '0.7(7.0)'], dtype=object)}
       
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

# prior = {}
# prior['m_{kplus,0}'] = gv.gvar(0.35,.1)
# prior['m_{eta,0}'] = gv.gvar(.3,.2)
# prior['m_{piplus,0}'] = gv.gvar(.25,.1)
# prior['m_{delta,0}'] = gv.gvar(2,1)

# TODO put prior routines in here, filename save options 
priors = gv.BufferDict()
