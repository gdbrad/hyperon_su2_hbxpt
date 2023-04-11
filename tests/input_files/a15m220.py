import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a15m220', #CHANGE THIS
    'hyperons' : ['delta_pp', 'lambda_z', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'meson_states' : ['piplus','kplus'],
    'simult_baryons': ['sigma_p','lambda_z','proton','xi_z'],
    'srcs'     :['S'],
    'snks'     :['SS','PS'],
    'bs_seed' : 'a12m400',

   't_range' : {
        'sigma' : [5,15],
        'sigma_st' : [5,15],
        'xi' :  [5,15],
        'xi_st' : [5,15],
        'proton' :   [5,15],
        'delta' : [5,15],
        'lam' : [5,15],
        'pi' : [5,30],
        'kplus': [8,28],
        'hyperons':   [5,15],
        'all':   [5,15],


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

    'tag':{
        'sigma' : 'sigma',
        'sigma_st' : 'sigma_st',
        'xi' :  'xi',
        'xi_st' : 'xi_st',
        'lam' : 'lam',
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}
prior = gv.BufferDict()
prior = {
    'sigma_E': np.array(['0.83(22)', '0.9(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.95(22)', '0.99(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_st_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'lam_E': np.array(['0.7(2.2)', '0.9(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'proton_E': np.array(['0.71(22)', '0.9(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'proton_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'delta_E': np.array(['0.91(22)', '0.95(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'delta_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'xi_E': np.array(['0.8(2.2)', '1.28(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.95(2.2)', '1.28(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_st_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    }


# prior = {}
# prior['m_{kplus,0}'] = gv.gvar(0.35,.1)
# prior['m_{eta,0}'] = gv.gvar(.3,.2)
# prior['m_{piplus,0}'] = gv.gvar(.25,.1)
# prior['m_{delta,0}'] = gv.gvar(2,1)

# TODO put prior routines in here, filename save options 
