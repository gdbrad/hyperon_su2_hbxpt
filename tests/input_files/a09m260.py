import gvar as gv 
import numpy as np 
p_dict = {
    'abbr': 'a09m260',
    'hyperons' : ['delta_pp', 'lambda_z', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'meson_states' : ['piplus','kplus'],
    'simult_baryons': ['sigma_p','lambda_z','proton','xi_z'],
    'srcs'     :['S'],
    'snks'     :['SS','PS'],
    'bs_seed' : 'a09m135',

   't_range' : {
        'sigma' : [7,15],
        'sigma_st' : [7,15],
        'xi' :  [7,15],
        'xi_st' : [8,15],
        'lam' : [7,15],
        'pi' : [5,30],
        'kplus': [8,28],
        'hyperons':   [7,15],
        'all':   [7,15],


    },
    'tag':{
            'sigma' : 'sigma',
            'sigma_st' : 'sigma_st',
            'xi' :  'xi',
            'xi_st' : 'xi_st',
            'lam' : 'lam',
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
        'all':3
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

prior = gv.BufferDict()
prior = {
    'sigma_E': np.array(['0.63(2.2)', '0.7(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.75(2.2)', '0.89(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_st_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'lam_E': np.array(['0.6(2.2)', '0.75(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.6(2.2)', '0.9(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.75(2.2)', '0.98(32)', '1.45(32)', '1.55(32)'], dtype=object),
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
