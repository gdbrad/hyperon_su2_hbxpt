import gvar as gv 
import numpy as np 
p_dict = {
    'abbr': 'a09m400',
    'hyperons' : ['delta_pp', 'lambda_z', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'meson_states' : ['piplus','kplus'],
    'simulbaryons': ['sigma_p','lambda_z','proton','xi_z'],
    'srcs'     :['S'],
    'snks'     :['SS','PS'],
    'bs_seed' : 'a09m135',

    'tag':{
        'sigma' : 'sigma',
        'sigma_st' : 'sigma_st',
        'xi' :  'xi',
        'xi_st' : 'xi_st',
        'lam' : 'lam'
},

    't_range':{

    'sigma' : [6,18],
    'sigma_st' : [8,17],
    'xi' :  [6,18],
    'xi_st' : [8,15],
    'proton' :   [8,17],
    'delta' : [10,19],
    'lam' : [6,18],
    'pi' : [5,30],
    'kplus': [8,28],
    'hyperons':   [6,18],
    'all':   [6,18],

    },
    'n_states' : {
        'sigma' : 2,
        'sigma_st' : 2,
        'xi' :2,
        'xi_st' :2,
        'delta':2,
        'proton':2,
        'lam':2,
        'pi' : 2,
        'kplus': 2,
        'mesons':2,
	    'hyperons'   :2,
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
    'sigma_E': np.array(['0.63(22)', '0.7(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.75(22)', '0.89(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_st_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'lam_E': np.array(['0.6(22)', '0.75(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'proton_E': np.array(['0.51(22)', '0.7(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'proton_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'delta_E': np.array(['0.6(22)', '0.95(22)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'delta_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'xi_E': np.array(['0.6(22)', '0.9(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.75(22)', '0.98(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_st_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    }
# '''
# $\delta_{GMO}$ xpt extrapolation model and prior information
# '''
# model_info = {}
# model_info['particles'] = ['piplus','kplus','eta']
# model_info['order_chiral'] = 'lo'
# model_info['tree_level'] = True
# model_info['loop_level'] = False
# model_info['delta'] = True
# model_info['abbr'] = ['a12m180L']
# model_info['observable'] = ['delta_gmo'] #'centroid', 'octet'

# prior = {}
# prior['m_{kplus,0}'] = gv.gvar(0.35,.1)
# prior['m_{eta,0}'] = gv.gvar(.3,.2)
# prior['m_{piplus,0}'] = gv.gvar(.25,.1)
# prior['m_{delta,0}'] = gv.gvar(2,1)

# # TODO put prior routines in here, filename save options 
# priors = gv.BufferDict()
